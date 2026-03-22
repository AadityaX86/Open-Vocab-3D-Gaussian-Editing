import torch
import numpy as np
import faiss
import os
import open_clip
import viser
import viser.transforms as tf
import time
import matplotlib
import matplotlib.cm as cm
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args

import open3d as o3d

# =========================================================================================
#  MATH & GEOMETRY HELPERS
# =========================================================================================

def refine_mask(
    gaussians,
    initial_mask,
    similarity_scores,
    eps_multiplier=4.0,
    min_points=5,
    min_cluster_size=50,
):
    """
    Robust Refinement using Median Nearest-Neighbor Density.
    Ignores extreme outliers to find the true object scale.
    """
    if not torch.any(initial_mask):
        return initial_mask

    # 1. MOVE TO CPU
    selected_indices = torch.where(initial_mask)[0]
    xyz_selected = gaussians._xyz[selected_indices].detach().cpu().numpy()
    
    # If the selection is tiny, don't bother clustering
    if len(xyz_selected) < 50:
        return initial_mask

    # 2. CREATE POINT CLOUD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_selected)
    
    # 3. CALCULATE LOCAL DENSITY (The Magic Step)
    # Measure distance from every point to its closest neighbor
    distances = pcd.compute_nearest_neighbor_distance()
    
    # Use the Median to ignore outliers completely
    median_dist = np.median(distances)
    
    # 4. SET DYNAMIC EPS AND VOXEL SIZE
    # Voxel size cleans up redundant points to save memory
    voxel_size = max(median_dist, 0.001) # Prevent 0
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # eps is usually 3x to 5x the median distance for good clustering
    dynamic_eps = voxel_size * max(eps_multiplier, 0.1)
    
    # 5. OPEN3D DBSCAN
    # min_points is low because we already voxel downsampled
    labels = np.array(downpcd.cluster_dbscan(eps=dynamic_eps, min_points=max(int(min_points), 1)))
    
    if len(labels) == 0 or np.all(labels == -1):
        return initial_mask

    # 6. IDENTIFY THE CORE OBJECT
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) == 0:
        return initial_mask
        
    main_label = unique_labels[np.argmax(counts)]
    
    # 7. SPATIAL FILTERING (Keep original points near the core cluster)
    main_cluster_points = np.asarray(downpcd.points)[labels == main_label]
    
    # Get bounding box of JUST the core object
    c_min = main_cluster_points.min(axis=0) - dynamic_eps
    c_max = main_cluster_points.max(axis=0) + dynamic_eps
    
    # Filter the original indices based on the clean core bounds
    all_xyz = gaussians._xyz[initial_mask].cpu()
    in_bounds = (all_xyz >= torch.from_numpy(c_min)) & (all_xyz <= torch.from_numpy(c_max))
    spatial_mask = in_bounds.all(dim=1)
    
    refined_indices = selected_indices[spatial_mask]
    if refined_indices.numel() < int(min_cluster_size):
        return initial_mask
    
    new_mask = torch.zeros_like(initial_mask)
    new_mask[refined_indices] = True
    return new_mask

def rotate_selection(gaussians, mask, roll_pitch_yaw):
    """
    Rotates the selection around its center of mass.
    """
    with torch.no_grad():
        # 1. Calculate Center of Mass of the selection
        # We rotate points AROUND this center, not the world origin (0,0,0)
        center = gaussians._xyz[mask].mean(dim=0)
        
        # 2. Create Rotation Matrix using Viser's helper
        rot_obj = tf.SO3.from_rpy_radians(*roll_pitch_yaw)
        R = torch.tensor(rot_obj.as_matrix(), device=gaussians._xyz.device, dtype=torch.float32)
        
        # 3. Rotate Positions: P_new = R @ (P_old - Center) + Center
        xyz_centered = gaussians._xyz[mask] - center
        gaussians._xyz[mask] = (xyz_centered @ R.T) + center
        
        # 4. Rotate Local Orientations (Quaternions)
        # We must update the orientation of the splats themselves, otherwise 
        # a "flat" splat will stay flat while moving in a circle.
        q_new = torch.tensor(rot_obj.wxyz, device=gaussians._rotation.device, dtype=torch.float32)        
        
        def quat_mult(q1, q2):
            # Batched quaternion multiplication (w, x, y, z)
            # This implements the Hamilton product
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)

        # Apply rotation: q_final = q_rotation * q_original
        gaussians._rotation[mask] = quat_mult(q_new, gaussians._rotation[mask])
        
        # Re-normalize quaternions to avoid numerical drift over many rotations
        gaussians._rotation[mask] = torch.nn.functional.normalize(gaussians._rotation[mask], dim=-1)

def scale_selection(gaussians, mask, scale_factor):
    """
    Scales the selection relative to its center of mass.
    """
    if scale_factor <= 0: return
    
    with torch.no_grad():
        # 1. Calculate Center
        center = gaussians._xyz[mask].mean(dim=0)
        
        # 2. Scale relative positions: P' = S * (P - C) + C
        # Moves points further apart (expansion) or closer together (contraction)
        gaussians._xyz[mask] = (gaussians._xyz[mask] - center) * scale_factor + center
        
        # 3. Scale the size of the Splats themselves
        # Gaussian splatting stores scale in Log-Space for stability.
        # log(scale * factor) = log(scale) + log(factor)
        gaussians._scaling[mask] += np.log(scale_factor)

def delete_selection(gaussians, mask):
    """
    'Deletes' objects by setting opacity to -infinity.
    """
    with torch.no_grad():
        # -100.0 becomes ~0.0 opacity after Sigmoid activation
        gaussians._opacity[mask] = -100.0

def color_selection(gaussians, mask, target_rgb):
    """
    Paints selected objects by overwriting the 0th Spherical Harmonic.
    """
    SH_C0 = 0.28209479177387814
    target_sh = (target_rgb - 0.5) / SH_C0
    
    with torch.no_grad():
        new_color = torch.tensor(target_sh, device=gaussians._features_dc.device).float()
        gaussians._features_dc[mask, 0, :] = new_color

def move_selection(gaussians, mask, offset_vector):
    """
    Translates selected objects in 3D space.
    """
    with torch.no_grad():
        offset = torch.tensor(offset_vector, device=gaussians._xyz.device).float()
        gaussians._xyz[mask] += offset

def get_colormap_safe(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)

def apply_overlay_heatmap(similarities, base_rgbs, threshold=0.2, colormap_name='turbo'):
    """
    Overlays heatmap on matches, keeps original color for background.
    """
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
    max_sim = similarities.max()
    
    if max_sim <= threshold:
        return base_rgbs 

    # Normalize (Absolute Scale: 0.0=Blue, Max=Red)
    # Added 1e-8 for safety
    norm_sim = similarities / (max_sim + 1e-8)
    norm_sim = torch.clamp(norm_sim, 0, 1)

    cmap = get_colormap_safe(colormap_name)
    heatmap_colors_cpu = cmap(norm_sim.cpu().numpy())[:, :3]
    heatmap_colors = torch.tensor(heatmap_colors_cpu, device=base_rgbs.device, dtype=torch.float32)
    
    mask = (similarities > threshold).float().unsqueeze(-1).to(base_rgbs.device)
    
    final_colors = heatmap_colors * mask + base_rgbs * (1 - mask)
    return final_colors

# =========================================================================================
#  MAIN APPLICATION
# =========================================================================================

def main(args, dataset_args, pipeline_args):
    # --- 1. SETUP & LOAD MODELS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print(f"\n[1/5] Loading OpenCLIP Model (ViT-B-16)...")
    model_name = "ViT-B-16" 
    pretrained_source = "laion2b_s34b_b88k"
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_source, device="cpu"
    )
    clip_model.eval() # Set to evaluation mode
    tokenizer = open_clip.get_tokenizer(model_name)
    
    print(f"[2/5] Loading Gaussian Model from {args.model_path}...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    checkpoint_path = os.path.join(args.model_path, "chkpnt0.pth") 
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, args, mode='test')

    # --- BACKUP ORIGINAL STATE (For Resetting) ---
    # CRITICAL IMPROVEMENT: Backup Rotation and Scaling too!
    # Without this, "Reset" would leave objects rotated/scaled incorrectly.
    print("Backing up initial scene state...")
    orig_features_dc = gaussians._features_dc.clone()
    orig_opacity = gaussians._opacity.clone()
    orig_xyz = gaussians._xyz.clone()
    orig_rotation = gaussians._rotation.clone() # Added backup
    orig_scaling = gaussians._scaling.clone()   # Added backup
    
    # --- 2. DECODE LANGUAGE FEATURES ---
    print("[3/5] Loading FAISS Index and Decoding Features...")
    index = faiss.read_index(args.pq_index)
    
    language_features_idx = gaussians._language_feature.clone()
    check_valid = torch.sum(language_features_idx, 1)
    invalid_index = check_valid == 255 * (index.coarse_code_size() + index.code_size)
    
    decoded_features_cpu = np.zeros((language_features_idx.shape[0], 512), dtype=np.float32)
    valid_mask_cpu = (invalid_index.cpu() == False).numpy()
    
    decoded_features_cpu[valid_mask_cpu] = index.sa_decode(
        language_features_idx[valid_mask_cpu].cpu().numpy()
    )
    
    # FP16 Optimization: Reduces VRAM usage by 50%
    gaussian_features = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float16)
    del decoded_features_cpu
    torch.cuda.empty_cache()
    
    # Normalization (FP16 Safe)
    norm = gaussian_features.norm(dim=-1, keepdim=True)
    # IMPROVEMENT: 1e-5 is safer than 1e-9 for float16
    gaussian_features.div_(norm + 1e-5)

    # --- 3. START VISER SERVER ---
    print("[4/5] Starting Viser Server...")
    server = viser.ViserServer(port=args.port)
    
    # State dictionary
    state = {
        "current_mask": None,
        "last_query": "",
        "last_similarities": None,
        "last_render_time": 0.0,
        "undo_snapshot": None,
    }

    render_throttle_s = 0.08

    # --- GUI LAYOUT ---
    with server.gui.add_folder("Semantic Editor"):
        # Search
        gui_query = server.gui.add_text("Target Object", initial_value="bicycle")
        gui_negative_query = server.gui.add_text("Negative Prompt", initial_value="floor, wall, background, blurry, dark, ground")
        gui_render_time = server.gui.add_text("Render Time (ms)", initial_value="0.0 ms", disabled=True)
        gui_max_scale = server.gui.add_slider("Max Splat Scale (Culling)", min=0.01, max=0.55, step=0.01, initial_value=0.55)
        
        gui_threshold = server.gui.add_slider(
            "Selection Threshold", 
            min=0.0, max=1.0, step=0.01, initial_value=0.22
        )

        gui_use_topk = server.gui.add_checkbox("Use Top-K Selection", initial_value=False)
        gui_topk_percent = server.gui.add_slider(
            "Top-K Percent",
            min=0.1,
            max=100.0,
            step=0.1,
            initial_value=5.0,
        )

        gui_selection_opacity_min = server.gui.add_slider(
            "Selection Opacity Min",
            min=0.0,
            max=0.5,
            step=0.01,
            initial_value=0.05,
        )

        btn_reset_all = server.gui.add_button("Reset All Edits", color="red")
        
        # Paint / Delete
        gui_color_picker = server.gui.add_rgb("Paint Color", initial_value=(1.0, 0.0, 0.0))
        btn_paint = server.gui.add_button("Apply Paint")
        btn_delete = server.gui.add_button("Delete Selection")
        btn_hard_delete = server.gui.add_button("Hard Delete (Free VRAM)", color="red")
        
        # Transformations
        with server.gui.add_folder("Transform"):
            with server.gui.add_folder("Translation"):
                slider_x = server.gui.add_slider("X Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
                slider_y = server.gui.add_slider("Y Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
                slider_z = server.gui.add_slider("Z Offset", min=-5.0, max=5.0, step=0.1, initial_value=0.0)
                btn_move = server.gui.add_button("Apply Move", color="cyan")

            with server.gui.add_folder("Rotation"):
                slider_roll = server.gui.add_slider("Roll (rad)", min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                slider_pitch = server.gui.add_slider("Pitch (rad)", min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                slider_yaw = server.gui.add_slider("Yaw (rad)", min=-3.14, max=3.14, step=0.01, initial_value=0.0)
                btn_rotate = server.gui.add_button("Apply Rotation", color="cyan")

            with server.gui.add_folder("Scaling"):
                slider_scale = server.gui.add_slider("Scale Factor", min=0.1, max=5.0, step=0.1, initial_value=1.0)
                btn_scale = server.gui.add_button("Apply Scale", color="cyan")

        gui_status = server.gui.add_text("Status", initial_value="Ready", disabled=True)

        btn_refine = server.gui.add_button("Refine Selection (DBSCAN)", color="green")
        gui_refine_eps_mult = server.gui.add_slider(
            "Refine Eps Multiplier",
            min=1.0,
            max=10.0,
            step=0.1,
            initial_value=4.0,
        )
        gui_refine_min_points = server.gui.add_slider(
            "Refine Min Points",
            min=1,
            max=30,
            step=1,
            initial_value=5,
        )
        gui_refine_min_cluster = server.gui.add_slider(
            "Refine Min Cluster Size",
            min=1,
            max=500,
            step=1,
            initial_value=50,
        )

        btn_undo = server.gui.add_button("Undo Last Edit", color="orange")

    # --- CORE RENDER LOOP ---
    def render_scene(highlight_mask=None, similarity_scores=None, force=False):
        """Fetches Gaussian state, culls invisible points, sends to Viser."""
        import time
        start_time = time.perf_counter()
        
        now = time.time()
        if not force and (now - state["last_render_time"]) < render_throttle_s:
            return
        state["last_render_time"] = now
        
        # 1. Get Opacities & Scales for GPU Culling
        opacities = gaussians.get_opacity
        scales_t = gaussians.get_scaling
        max_scales = scales_t.max(dim=-1).values
        
        print(f"Scales Mean: {max_scales.mean().item():.4f}, Max: {max_scales.max().item():.4f}")
        # 2. Smart Culling: GPU-accelerated mask (Opacity + Scale Culling)
        # Fixes ghosting artifacts, sorting errors, and saves VRAM memory bandwidth
        vis_mask_tensor = (opacities.squeeze() > 0.10) & (max_scales < gui_max_scale.value)
        
        if not vis_mask_tensor.any():
            return 

        # 3. Extract & Filter Geometry natively on GPU to save PCIe bandwidth
        means = gaussians.get_xyz[vis_mask_tensor].detach().cpu().numpy()
        opacities_vis = opacities[vis_mask_tensor].detach().cpu().numpy()
        scales_gpu = scales_t[vis_mask_tensor].detach()
        quats_gpu = gaussians.get_rotation[vis_mask_tensor].detach()
        
        # 4. Compute Covariances on GPU (PyTorch > NumPy/einsum)
        Rs_cpu = tf.SO3(quats_gpu.cpu().numpy()).as_matrix()
        Rs_gpu = torch.tensor(Rs_cpu, device=scales_gpu.device, dtype=torch.float32)
        
        S = torch.zeros((scales_gpu.shape[0], 3, 3), device=scales_gpu.device, dtype=torch.float32)
        S[:, 0, 0] = scales_gpu[:, 0]
        S[:, 1, 1] = scales_gpu[:, 1]
        S[:, 2, 2] = scales_gpu[:, 2]
        
        M = torch.bmm(Rs_gpu, S)
        covariances_gpu = torch.bmm(M, M.transpose(1, 2))
        covariances = covariances_gpu.cpu().numpy()

        # 5. Extract Colors & Overlay Heatmap
        shs = gaussians.get_features
        base_rgbs_gpu = (shs[:, 0, :].detach() * 0.28209479177387814 + 0.5).clamp(0, 1)

        final_rgbs_gpu = base_rgbs_gpu
        if similarity_scores is not None and highlight_mask is not None:
             final_rgbs_gpu = apply_overlay_heatmap(
                similarity_scores, 
                base_rgbs_gpu, 
                threshold=gui_threshold.value
            )
        
        rgbs_vis = final_rgbs_gpu[vis_mask_tensor].cpu().numpy()

        # 6. Send to Viser
        server.scene.add_gaussian_splats(
            "/scene",
            centers=means,
            rgbs=rgbs_vis,
            opacities=opacities_vis,
            covariances=covariances
        )
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        gui_render_time.value = f"{elapsed_ms:.1f} ms"

    # --- CALLBACKS ---

    def compute_visibility_mask():
        return gaussians.get_opacity.squeeze() > gui_selection_opacity_min.value

    def build_selection_mask(similarity):
        if similarity is None:
            return None

        if gui_use_topk.value:
            n = similarity.shape[0]
            k = max(1, int(n * (gui_topk_percent.value / 100.0)))
            cutoff = torch.topk(similarity, k, largest=True, sorted=True).values[-1]
            semantic_mask = similarity >= cutoff
        else:
            semantic_mask = similarity > gui_threshold.value

        visible_mask = compute_visibility_mask().to(semantic_mask.device)
        return semantic_mask & visible_mask

    def apply_selection_from_cache(force_render=False):
        if state["last_similarities"] is None:
            return

        similarity = state["last_similarities"]
        state["current_mask"] = build_selection_mask(similarity)

        max_sim = similarity.max().item()
        selected = int(state["current_mask"].sum().item()) if state["current_mask"] is not None else 0
        gui_status.value = f"Max: {max_sim:.4f} | Selected: {selected}"
        render_scene(
            highlight_mask=state["current_mask"],
            similarity_scores=similarity,
            force=force_render,
        )

    def save_undo_snapshot(mask, fields):
        indices = torch.where(mask)[0]
        if indices.numel() == 0:
            return

        snapshot = {"indices": indices, "fields": {}}
        with torch.no_grad():
            if "xyz" in fields:
                snapshot["fields"]["xyz"] = gaussians._xyz[indices].clone()
            if "rotation" in fields:
                snapshot["fields"]["rotation"] = gaussians._rotation[indices].clone()
            if "scaling" in fields:
                snapshot["fields"]["scaling"] = gaussians._scaling[indices].clone()
            if "opacity" in fields:
                snapshot["fields"]["opacity"] = gaussians._opacity[indices].clone()
            if "features_dc" in fields:
                snapshot["fields"]["features_dc"] = gaussians._features_dc[indices].clone()

        state["undo_snapshot"] = snapshot

    def handle_refine(_):
        if state["current_mask"] is None:
            return
        gui_status.value = "Refining clusters..."
        eps_multiplier = gui_refine_eps_mult.value
        min_points = int(gui_refine_min_points.value)
        min_cluster_size = int(gui_refine_min_cluster.value)
        
        # Use the similarity scores stored in the state/calculated earlier
        state["current_mask"] = refine_mask(
            gaussians,
            state["current_mask"],
            state["last_similarities"],
            eps_multiplier=eps_multiplier,
            min_points=min_points,
            min_cluster_size=min_cluster_size,
        )
        
        selected = int(state["current_mask"].sum().item())
        gui_status.value = f"Refined! Selected: {selected}"
        render_scene(highlight_mask=state["current_mask"], similarity_scores=state["last_similarities"], force=True)


    def update_query_selection(_):
        query_text = gui_query.value
        negative_query_text = gui_negative_query.value
        if not query_text:
            return
            
        current_state_str = query_text + "|" + negative_query_text
        if current_state_str == state.get("last_query_state") and state["last_similarities"] is not None:
            apply_selection_from_cache(force_render=True)
            return

        gui_status.value = f"Processing '{query_text}'..."
        
        with torch.no_grad():
            # 1. ENCODE POSITIVE WITH TEMPLATES ON CPU
            templates = [
                f"a 3d render of a {query_text}",
                f"a clear photo of a {query_text}",
                f"a single {query_text}",
                f"a close-up of a {query_text}",
                f"a {query_text} in a scene",
            ]
            pos_tokens = tokenizer(templates).to("cpu")
            pos_features = clip_model.encode_text(pos_tokens).mean(dim=0, keepdim=True)
            
            # ENCODE NEGATIVE ON CPU
            neg_tokens = tokenizer([negative_query_text]).to("cpu")
            neg_features = clip_model.encode_text(neg_tokens)
            
            # 2. MATCH DTYPE & MOVE TO GPU
            pos_features = pos_features.to(device=gaussian_features.device, dtype=gaussian_features.dtype)
            pos_features /= pos_features.norm(dim=-1, keepdim=True)
            
            neg_features = neg_features.to(device=gaussian_features.device, dtype=gaussian_features.dtype)
            neg_features /= neg_features.norm(dim=-1, keepdim=True)
            
            # 3. CONTRASTIVE SIMILARITY
            sim_pos = (gaussian_features @ pos_features.T).squeeze()
            sim_neg = (gaussian_features @ neg_features.T).squeeze()
            
            similarity = sim_pos - (0.5 * sim_neg)
            similarity = torch.nan_to_num(similarity, nan=0.0)

            state["last_similarities"] = similarity
            state["last_query"] = query_text
            state["last_query_state"] = current_state_str

        apply_selection_from_cache(force_render=True)

    def update_threshold_only(_):
        apply_selection_from_cache(force_render=False)

    def handle_undo(_):
        snapshot = state["undo_snapshot"]
        if snapshot is None:
            gui_status.value = "Undo empty"
            return

        idx = snapshot["indices"]
        fields = snapshot["fields"]
        with torch.no_grad():
            if "xyz" in fields:
                gaussians._xyz[idx] = fields["xyz"]
            if "rotation" in fields:
                gaussians._rotation[idx] = fields["rotation"]
            if "scaling" in fields:
                gaussians._scaling[idx] = fields["scaling"]
            if "opacity" in fields:
                gaussians._opacity[idx] = fields["opacity"]
            if "features_dc" in fields:
                gaussians._features_dc[idx] = fields["features_dc"]

        state["undo_snapshot"] = None
        gui_status.value = "Undo applied"
        apply_selection_from_cache(force_render=True)

    def handle_rotate(_):
        if state["current_mask"] is None:
            return
        angles = [slider_roll.value, slider_pitch.value, slider_yaw.value]
        save_undo_snapshot(state["current_mask"], fields=["xyz", "rotation"])
        print(f"Rotating selection by {angles} radians...")
        rotate_selection(gaussians, state["current_mask"], angles)
        
        # Reset sliders to 0.0 (Relative Transform)
        slider_roll.value = 0.0
        slider_pitch.value = 0.0
        slider_yaw.value = 0.0
        apply_selection_from_cache(force_render=True)

    def handle_scale(_):
        if state["current_mask"] is None:
            return
        factor = slider_scale.value
        save_undo_snapshot(state["current_mask"], fields=["xyz", "scaling"])
        print(f"Scaling selection by {factor}x...")
        scale_selection(gaussians, state["current_mask"], factor)
        
        # Reset slider to 1.0 (Relative Transform)
        slider_scale.value = 1.0
        apply_selection_from_cache(force_render=True)

    def handle_delete(_):
        if state["current_mask"] is None:
            return
        save_undo_snapshot(state["current_mask"], fields=["opacity"])
        delete_selection(gaussians, state["current_mask"])
        apply_selection_from_cache(force_render=True)

    def handle_hard_delete(_):
        if state["current_mask"] is None:
            return
        # Keep everything that is NOT selected
        keep_mask = ~state["current_mask"]
        
        with torch.no_grad():
            gaussians._xyz = gaussians._xyz[keep_mask]
            gaussians._rotation = gaussians._rotation[keep_mask]
            gaussians._scaling = gaussians._scaling[keep_mask]
            gaussians._opacity = gaussians._opacity[keep_mask]
            gaussians._features_dc = gaussians._features_dc[keep_mask]
            gaussians._features_rest = gaussians._features_rest[keep_mask]
            gaussians._language_feature = gaussians._language_feature[keep_mask] # also slice language features just in case
            
            nonlocal gaussian_features
            gaussian_features = gaussian_features[keep_mask]

        state["current_mask"] = None
        state["last_similarities"] = None
        state["undo_snapshot"] = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        gui_status.value = "Hard Delete Applied. VRAM Freed."
        render_scene(force=True)

    def handle_paint(_):
        if state["current_mask"] is None:
            return
        save_undo_snapshot(state["current_mask"], fields=["features_dc"])
        color_selection(gaussians, state["current_mask"], np.array(gui_color_picker.value))
        apply_selection_from_cache(force_render=True)
    
    def handle_move(_):
        if state["current_mask"] is None:
            return
        offset = [slider_x.value, slider_y.value, slider_z.value]
        save_undo_snapshot(state["current_mask"], fields=["xyz"])
        move_selection(gaussians, state["current_mask"], offset)
        
        slider_x.value = 0.0
        slider_y.value = 0.0
        slider_z.value = 0.0
        apply_selection_from_cache(force_render=True)

    def handle_reset_scene(_):
        """Restores the model to its initial state."""
        with torch.no_grad():
            # Full Restore: Position, Color, Opacity, Rotation, Scale
            gaussians._features_dc.copy_(orig_features_dc)
            gaussians._opacity.copy_(orig_opacity)
            gaussians._xyz.copy_(orig_xyz)
            gaussians._rotation.copy_(orig_rotation) # Added
            gaussians._scaling.copy_(orig_scaling)   # Added
            
        state["current_mask"] = None
        state["last_similarities"] = None
        state["undo_snapshot"] = None
        gui_query.value = ""
        gui_status.value = "Scene Reset"
        render_scene(highlight_mask=None, similarity_scores=None, force=True)
        print("Scene fully restored.")

    # --- BINDING ---
    gui_query.on_update(update_query_selection)
    gui_negative_query.on_update(update_query_selection)
    gui_threshold.on_update(update_threshold_only)
    gui_use_topk.on_update(update_threshold_only)
    gui_topk_percent.on_update(update_threshold_only)
    gui_selection_opacity_min.on_update(update_threshold_only)
    
    btn_delete.on_click(handle_delete)
    btn_hard_delete.on_click(handle_hard_delete)
    btn_paint.on_click(handle_paint)
    btn_reset_all.on_click(handle_reset_scene)
    
    btn_move.on_click(handle_move)
    btn_rotate.on_click(handle_rotate)
    btn_scale.on_click(handle_scale)

    btn_refine.on_click(handle_refine)
    btn_undo.on_click(handle_undo)


    # Initial Render
    print("Pre-loading scene...")
    render_scene()
    if gui_query.value:
        update_query_selection(None)
    
    print(f"SUCCESS! Interactive Editor running on http://localhost:{args.port}")
    print(f"------------------------------------------------\n")
    
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive Semantic Editor")
    lp = ModelParams(parser)
    op = PipelineParams(parser)
    
    parser.add_argument("--query", type=str, default="bicycle", help="Initial query")
    parser.add_argument("--threshold", type=float, default=0.22, help="Similarity threshold") 
    parser.add_argument("--pq_index", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--port", type=int, default=8080, help="Viser server port")
    
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))