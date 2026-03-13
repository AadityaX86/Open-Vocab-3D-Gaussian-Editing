from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid


@dataclass
class InpaintingConfig:
    """Hyperparameters for the zero-training 3D semantic inpainting pipeline."""

    # Step 1: Boundary extraction and RANSAC plane fit
    min_boundary_points: int = 64
    boundary_radius_scale: float = 4.0
    ransac_hypotheses: int = 128
    ransac_inlier_distance: float = 0.03

    # Step 2: Semantic clustering (CLIP embedding space)
    max_semantic_clusters: int = 3
    kmeans_iterations: int = 12

    # Step 3: Source retrieval and cloning
    clone_multiplier: float = 1.2
    max_clone_points: int = 15000

    # Step 5: Edge blending
    blend_knn_k: int = 5
    blend_edge_fraction: float = 0.3
    blend_alpha: float = 0.6


class RealTimeGaussianInpainter:
    """
    Real-time, feed-forward, zero-training 3D Gaussian inpainting.

    This class operates directly on GaussianModel tensors and never runs
    gradient descent, backpropagation, diffusion, or 2D refinement.
    """

    def __init__(self, gaussians: GaussianModel, pq_index, config: Optional[InpaintingConfig] = None):
        self.gaussians = gaussians
        self.pq_index = pq_index
        self.cfg = config or InpaintingConfig()

    @staticmethod
    def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return v / (torch.norm(v, dim=-1, keepdim=True) + eps)

    @staticmethod
    def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product for quaternions in (w, x, y, z) format."""
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    @classmethod
    def _quat_conjugate(cls, q: torch.Tensor) -> torch.Tensor:
        out = q.clone()
        out[..., 1:] = -out[..., 1:]
        return out

    @classmethod
    def _quat_rotate_points(cls, q: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Rotate 3D points using quaternion q (w, x, y, z)."""
        if pts.numel() == 0:
            return pts
        qn = cls._normalize(q)
        if qn.ndim == 1:
            qn = qn.unsqueeze(0)
        qn = qn.expand(pts.shape[0], 4)

        v_quat = torch.zeros((pts.shape[0], 4), device=pts.device, dtype=pts.dtype)
        v_quat[:, 1:] = pts
        rotated = cls._quat_mul(cls._quat_mul(qn, v_quat), cls._quat_conjugate(qn))
        return rotated[:, 1:]

    @classmethod
    def _quat_between_normals(cls, src_n: torch.Tensor, dst_n: torch.Tensor) -> torch.Tensor:
        """
        Quaternion aligning src_n to dst_n.

        For unit vectors a and b, quaternion q = [1 + dot(a,b), cross(a,b)]
        (followed by normalization), with anti-parallel fallback.
        """
        a = cls._normalize(src_n.view(1, 3))[0]
        b = cls._normalize(dst_n.view(1, 3))[0]
        dot_ab = torch.clamp(torch.dot(a, b), -1.0, 1.0)

        if dot_ab < -0.9999:
            # Anti-parallel case: choose an arbitrary orthogonal axis.
            axis = torch.tensor([1.0, 0.0, 0.0], device=a.device, dtype=a.dtype)
            if torch.abs(torch.dot(axis, a)) > 0.9:
                axis = torch.tensor([0.0, 1.0, 0.0], device=a.device, dtype=a.dtype)
            axis = cls._normalize(torch.cross(a, axis, dim=0).view(1, 3))[0]
            return torch.tensor([0.0, axis[0], axis[1], axis[2]], device=a.device, dtype=a.dtype)

        cross_ab = torch.cross(a, b, dim=0)
        q = torch.tensor([1.0 + dot_ab, cross_ab[0], cross_ab[1], cross_ab[2]], device=a.device, dtype=a.dtype)
        return cls._normalize(q.view(1, 4))[0]

    @staticmethod
    def _project_points_to_plane(points: torch.Tensor, plane_normal: torch.Tensor, plane_offset: torch.Tensor) -> torch.Tensor:
        """
        Orthogonally project points to plane n^T x + d = 0.

        Projection formula:
            x_proj = x - (n^T x + d) n
        with unit normal n.
        """
        n = plane_normal.view(1, 3)
        signed_dist = (points * n).sum(dim=-1, keepdim=True) + plane_offset
        return points - signed_dist * n

    @staticmethod
    def _nearest_distances_chunked(query: torch.Tensor, refs: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """Memory-safe nearest-neighbor distances using chunked cdist."""
        out = torch.full((query.shape[0],), float("inf"), device=query.device, dtype=query.dtype)
        for i in range(0, query.shape[0], chunk):
            qi = query[i : i + chunk]
            d = torch.cdist(qi, refs)
            out[i : i + chunk] = d.min(dim=1).values
        return out

    def _extract_boundary_indices(self, removed_mask: torch.Tensor) -> Tuple[torch.Tensor, float]:
        xyz = self.gaussians._xyz
        removed_idx = torch.where(removed_mask)[0]
        valid_idx = torch.where(~removed_mask)[0]

        if removed_idx.numel() == 0:
            return torch.empty(0, device=xyz.device, dtype=torch.long), 0.0

        removed_xyz = xyz[removed_idx]
        valid_xyz = xyz[valid_idx]

        # Radius based on removed Gaussian scale with bbox fallback.
        if self.gaussians._scaling.shape[0] > 0 and removed_idx.numel() > 0:
            local_scale = torch.exp(self.gaussians._scaling[removed_idx]).mean(dim=1)
            scale_radius = torch.median(local_scale).item() * self.cfg.boundary_radius_scale
        else:
            scale_radius = 0.0
        bbox_diag = torch.norm(removed_xyz.max(dim=0).values - removed_xyz.min(dim=0).values).item()
        boundary_radius = max(scale_radius, 0.01 * max(bbox_diag, 1e-4))

        min_dist = self._nearest_distances_chunked(valid_xyz, removed_xyz)
        boundary_local_mask = min_dist <= boundary_radius
        boundary_idx = valid_idx[boundary_local_mask]

        # Robust fallback: use closest points if radius-based selection is sparse.
        if boundary_idx.numel() < self.cfg.min_boundary_points:
            k = min(self.cfg.min_boundary_points, valid_idx.numel())
            topk = torch.topk(min_dist, k=k, largest=False).indices
            boundary_idx = valid_idx[topk]

        return boundary_idx, boundary_radius

    def _fit_plane_ransac(self, points: torch.Tensor, inlier_dist: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Robust plane fit with RANSAC.

        Returns:
            normal: unit normal vector, shape (3,)
            offset: scalar d in plane equation n^T x + d = 0
            inliers: bool mask over points
        """
        n_points = points.shape[0]
        if n_points < 3:
            raise ValueError("At least 3 points are required for plane fitting.")

        best_inliers = None
        best_count = -1

        for _ in range(self.cfg.ransac_hypotheses):
            sample_ids = torch.randperm(n_points, device=points.device)[:3]
            p0, p1, p2 = points[sample_ids]

            n = torch.cross(p1 - p0, p2 - p0, dim=0)
            n_norm = torch.norm(n)
            if n_norm < 1e-6:
                continue
            n = n / n_norm
            d = -torch.dot(n, p0)

            dist = torch.abs((points * n.view(1, 3)).sum(dim=1) + d)
            inliers = dist < inlier_dist
            count = int(inliers.sum().item())

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_inliers.sum() < 3:
            # Fallback to least-squares plane over all points.
            best_inliers = torch.ones((n_points,), dtype=torch.bool, device=points.device)

        inlier_pts = points[best_inliers]
        center = inlier_pts.mean(dim=0, keepdim=True)
        demeaned = inlier_pts - center
        _, _, vh = torch.linalg.svd(demeaned, full_matrices=False)
        normal = vh[-1]
        normal = self._normalize(normal.view(1, 3))[0]
        offset = -torch.dot(normal, center[0])
        return normal, offset, best_inliers

    def _decode_embeddings(self, pq_codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode PQ byte-codes to CLIP embeddings and return valid mask."""
        device = pq_codes.device
        n = pq_codes.shape[0]
        code_len = pq_codes.shape[1]
        invalid_sum = 255 * code_len
        valid = pq_codes.sum(dim=1) != invalid_sum

        decoded = np.zeros((n, 512), dtype=np.float32)
        if valid.any():
            decoded_valid = self.pq_index.sa_decode(pq_codes[valid].detach().cpu().numpy())
            decoded[valid.detach().cpu().numpy()] = decoded_valid

        emb = torch.tensor(decoded, device=device, dtype=torch.float32)
        emb = self._normalize(emb)
        return emb, valid

    def _kmeans_torch(self, x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple GPU K-Means for small boundary embedding sets."""
        n = x.shape[0]
        if n == 0:
            raise ValueError("K-Means received no samples.")

        k = max(1, min(k, n))
        init_ids = torch.randperm(n, device=x.device)[:k]
        centers = x[init_ids].clone()

        for _ in range(self.cfg.kmeans_iterations):
            d = torch.cdist(x, centers)
            labels = torch.argmin(d, dim=1)
            for ci in range(k):
                mask = labels == ci
                if mask.any():
                    centers[ci] = x[mask].mean(dim=0)

        centers = self._normalize(centers)
        d = torch.cdist(x, centers)
        labels = torch.argmin(d, dim=1)
        return centers, labels

    def _map_centers_to_pq_codes(self, centers: torch.Tensor, boundary_codes: torch.Tensor, boundary_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map semantic cluster centers to nearest PQ centroids.

        Preferred route is direct SA encoding. If not available in index bindings,
        fallback to nearest decoded boundary embedding and reuse its code.
        """
        centers_np = centers.detach().cpu().numpy().astype(np.float32)

        if hasattr(self.pq_index, "sa_encode"):
            encoded = self.pq_index.sa_encode(centers_np)
            return torch.tensor(encoded, device=centers.device, dtype=torch.uint8)

        sim = torch.matmul(centers, boundary_embeddings.t())
        nn_ids = torch.argmax(sim, dim=1)
        return boundary_codes[nn_ids]

    def _append_gaussians(
        self,
        new_xyz: torch.Tensor,
        new_scaling: torch.Tensor,
        new_rotation: torch.Tensor,
        new_opacity: torch.Tensor,
        new_features_dc: torch.Tensor,
        new_features_rest: torch.Tensor,
        new_language_feature: Optional[torch.Tensor],
    ) -> Tuple[int, int]:
        """Append cloned gaussians to model tensors in test/edit mode."""
        old_n = self.gaussians._xyz.shape[0]

        def _cat_param(attr_name: str, ext: torch.Tensor) -> None:
            old_val = getattr(self.gaussians, attr_name)
            if isinstance(old_val, nn.Parameter):
                combined = torch.cat([old_val.data, ext], dim=0)
                setattr(self.gaussians, attr_name, nn.Parameter(combined, requires_grad=old_val.requires_grad))
            else:
                setattr(self.gaussians, attr_name, torch.cat([old_val, ext], dim=0))

        _cat_param("_xyz", new_xyz)
        _cat_param("_scaling", new_scaling)
        _cat_param("_rotation", new_rotation)
        _cat_param("_opacity", new_opacity)
        _cat_param("_features_dc", new_features_dc)
        _cat_param("_features_rest", new_features_rest)

        if new_language_feature is not None and self.gaussians._language_feature is not None:
            old_lf = self.gaussians._language_feature
            if isinstance(old_lf, nn.Parameter):
                lf_combined = torch.cat([old_lf.data, new_language_feature], dim=0)
                self.gaussians._language_feature = nn.Parameter(lf_combined, requires_grad=old_lf.requires_grad)
            else:
                self.gaussians._language_feature = torch.cat([old_lf, new_language_feature], dim=0)

        new_n = self.gaussians._xyz.shape[0]
        add_n = new_n - old_n

        # Keep auxiliary bookkeeping tensors dimensionally consistent.
        if self.gaussians.max_radii2D.numel() == old_n:
            self.gaussians.max_radii2D = torch.cat(
                [self.gaussians.max_radii2D, torch.zeros((add_n,), device=self.gaussians.max_radii2D.device)],
                dim=0,
            )
        if self.gaussians.xyz_gradient_accum.ndim == 2 and self.gaussians.xyz_gradient_accum.shape[0] == old_n:
            self.gaussians.xyz_gradient_accum = torch.cat(
                [
                    self.gaussians.xyz_gradient_accum,
                    torch.zeros((add_n, self.gaussians.xyz_gradient_accum.shape[1]), device=self.gaussians.xyz_gradient_accum.device),
                ],
                dim=0,
            )
        if self.gaussians.denom.ndim == 2 and self.gaussians.denom.shape[0] == old_n:
            self.gaussians.denom = torch.cat(
                [
                    self.gaussians.denom,
                    torch.zeros((add_n, self.gaussians.denom.shape[1]), device=self.gaussians.denom.device),
                ],
                dim=0,
            )

        return old_n, new_n

    def inpaint(self, removed_mask: torch.Tensor) -> Dict[str, int]:
        """
        Execute the 5-step real-time semantic patch inpainting pipeline.

        Args:
            removed_mask: bool tensor of shape (N,), True for removed gaussians.

        Returns:
            Stats dictionary with counts and status info.
        """
        with torch.no_grad():
            device = self.gaussians._xyz.device
            removed_mask = removed_mask.to(device=device, dtype=torch.bool)

            removed_idx = torch.where(removed_mask)[0]
            if removed_idx.numel() < 3:
                return {"status": 0, "reason": 1, "cloned": 0}

            # -----------------------------------------------------------------
            # Step 1: Void boundary extraction + RANSAC plane fit
            # -----------------------------------------------------------------
            boundary_idx, boundary_radius = self._extract_boundary_indices(removed_mask)
            if boundary_idx.numel() < 3:
                return {"status": 0, "reason": 2, "cloned": 0}

            boundary_xyz = self.gaussians._xyz[boundary_idx]
            plane_n, plane_d, _ = self._fit_plane_ransac(boundary_xyz, inlier_dist=self.cfg.ransac_inlier_distance)

            # -----------------------------------------------------------------
            # Step 2: Semantic boundary clustering + PQ centroid mapping
            # -----------------------------------------------------------------
            if self.gaussians._language_feature is not None:
                boundary_codes = self.gaussians._language_feature[boundary_idx]
                boundary_emb, valid_semantic = self._decode_embeddings(boundary_codes)
            else:
                boundary_codes = None
                boundary_emb = None
                valid_semantic = None

            pq_matched_centers = None
            if boundary_codes is not None and valid_semantic is not None and valid_semantic.any():
                semantic_emb = boundary_emb[valid_semantic]
                semantic_codes = boundary_codes[valid_semantic]
                k = min(self.cfg.max_semantic_clusters, semantic_emb.shape[0])
                centers, _ = self._kmeans_torch(semantic_emb, k=k)
                pq_matched_centers = self._map_centers_to_pq_codes(centers, semantic_codes, semantic_emb)

            # -----------------------------------------------------------------
            # Step 3: Source retrieval by matched PQ code + cloning
            # -----------------------------------------------------------------
            valid_idx = torch.where(~removed_mask)[0]
            all_xyz = self.gaussians._xyz

            if pq_matched_centers is not None and self.gaussians._language_feature is not None:
                all_codes = self.gaussians._language_feature[valid_idx]
                source_local_mask = torch.zeros((valid_idx.shape[0],), dtype=torch.bool, device=device)
                for ci in range(pq_matched_centers.shape[0]):
                    code = pq_matched_centers[ci].view(1, -1)
                    source_local_mask |= torch.all(all_codes == code, dim=1)
                source_idx = valid_idx[source_local_mask]
            else:
                source_idx = valid_idx

            if source_idx.numel() == 0:
                return {"status": 0, "reason": 3, "cloned": 0}

            # Prefer source points spatially close to the hole region for local texture continuity.
            removed_center = all_xyz[removed_idx].mean(dim=0, keepdim=True)
            source_xyz_all = all_xyz[source_idx]
            source_dist = torch.norm(source_xyz_all - removed_center, dim=1)

            target_clone_n = int(min(self.cfg.max_clone_points, max(32, int(removed_idx.numel() * self.cfg.clone_multiplier))))
            if source_idx.numel() >= target_clone_n:
                pick_local = torch.topk(source_dist, k=target_clone_n, largest=False).indices
                selected_source_idx = source_idx[pick_local]
            else:
                # If semantic retrieval is sparse, sample with replacement to fill the hole.
                rand_ids = torch.randint(0, source_idx.numel(), (target_clone_n,), device=device)
                selected_source_idx = source_idx[rand_ids]

            src_xyz = self.gaussians._xyz[selected_source_idx]
            src_scaling = self.gaussians._scaling[selected_source_idx].clone()
            src_rotation = self.gaussians._rotation[selected_source_idx].clone()
            src_opacity = self.gaussians._opacity[selected_source_idx].clone()
            src_fdc = self.gaussians._features_dc[selected_source_idx].clone()
            src_frest = self.gaussians._features_rest[selected_source_idx].clone()
            src_lf = self.gaussians._language_feature[selected_source_idx].clone() if self.gaussians._language_feature is not None else None

            # -----------------------------------------------------------------
            # Step 4: Placement + plane projection + quaternion orientation align
            # -----------------------------------------------------------------
            src_center = src_xyz.mean(dim=0, keepdim=True)

            # Source local plane estimated by PCA normal (smallest variance direction).
            src_demeaned = src_xyz - src_center
            _, _, src_vh = torch.linalg.svd(src_demeaned, full_matrices=False)
            src_plane_n = self._normalize(src_vh[-1].view(1, 3))[0]

            # Quaternion that rotates source-plane normal onto target (RANSAC) plane normal.
            q_align = self._quat_between_normals(src_plane_n, plane_n)

            # Rotate cloned point cloud around source centroid.
            src_local = src_xyz - src_center
            rot_local = self._quat_rotate_points(q_align, src_local)
            rotated_xyz = rot_local + src_center

            # Translate rotated patch toward hole center projected onto fitted plane.
            removed_center_proj = self._project_points_to_plane(removed_center, plane_n, plane_d)
            translation = removed_center_proj - rotated_xyz.mean(dim=0, keepdim=True)
            placed_xyz = rotated_xyz + translation

            # Final strict snap: project every cloned gaussian center exactly onto target plane.
            placed_xyz = self._project_points_to_plane(placed_xyz, plane_n, plane_d)

            # Compose orientation quaternions so Gaussian covariance orientation follows new surface.
            q_align_batched = q_align.view(1, 4).expand(src_rotation.shape[0], 4)
            placed_rotation = self._quat_mul(q_align_batched, src_rotation)
            placed_rotation = self._normalize(placed_rotation)

            # -----------------------------------------------------------------
            # Step 5: Seam blending for opacity + DC color only (KNN boundary)
            # -----------------------------------------------------------------
            if boundary_idx.numel() > 0:
                boundary_xyz = self.gaussians._xyz[boundary_idx]
                boundary_opacity_prob = torch.sigmoid(self.gaussians._opacity[boundary_idx])
                boundary_fdc = self.gaussians._features_dc[boundary_idx]

                full_dist = torch.cdist(placed_xyz, boundary_xyz)
                nearest_dist = full_dist.min(dim=1).values

                edge_count = max(1, int(self.cfg.blend_edge_fraction * placed_xyz.shape[0]))
                edge_ids = torch.topk(nearest_dist, k=edge_count, largest=False).indices

                k = min(self.cfg.blend_knn_k, boundary_idx.numel())
                edge_dist = full_dist[edge_ids]
                knn_dist, knn_ids = torch.topk(edge_dist, k=k, largest=False, dim=1)

                w = 1.0 / (knn_dist + 1e-6)
                w = w / w.sum(dim=1, keepdim=True)

                # Weighted neighborhood targets.
                neigh_opacity = boundary_opacity_prob[knn_ids]  # [E, K, 1]
                neigh_fdc = boundary_fdc[knn_ids]  # [E, K, 1, 3]

                target_opacity = (neigh_opacity * w.unsqueeze(-1)).sum(dim=1)
                target_fdc = (neigh_fdc * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

                # Blend only edge gaussians and only requested channels.
                alpha = self.cfg.blend_alpha
                placed_opacity_prob = torch.sigmoid(src_opacity)
                placed_opacity_prob[edge_ids] = (1.0 - alpha) * placed_opacity_prob[edge_ids] + alpha * target_opacity
                placed_fdc = src_fdc.clone()
                placed_fdc[edge_ids] = (1.0 - alpha) * placed_fdc[edge_ids] + alpha * target_fdc

                # Convert blended opacity back to logit parameter space expected by GaussianModel.
                placed_opacity = inverse_sigmoid(torch.clamp(placed_opacity_prob, 1e-4, 1.0 - 1e-4))
            else:
                placed_fdc = src_fdc
                placed_opacity = src_opacity

            # Commit new inpainted gaussians into model tensors.
            old_n, new_n = self._append_gaussians(
                new_xyz=placed_xyz,
                new_scaling=src_scaling,
                new_rotation=placed_rotation,
                new_opacity=placed_opacity,
                new_features_dc=placed_fdc,
                new_features_rest=src_frest,
                new_language_feature=src_lf,
            )

            return {
                "status": 1,
                "reason": 0,
                "removed": int(removed_idx.numel()),
                "boundary": int(boundary_idx.numel()),
                "cloned": int(new_n - old_n),
                "boundary_radius": int(boundary_radius * 1000),
            }