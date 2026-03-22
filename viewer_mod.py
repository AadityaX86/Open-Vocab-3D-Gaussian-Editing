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
from sklearn.cluster import DBSCAN  

# =========================================================================================
#  MATH & GEOMETRY HELPERS
# =========================================================================================

def refine_mask_dbscan(gaussians, mask, eps=0.1, min_samples=10):
    """
    Refines the selection by keeping only the largest spatial cluster.
    Uses DBSCAN to identify the main object and remove floating noise.
    """
    # optimization: return early if nothing selected
    if mask.sum() == 0: return mask

    # 1. Get coordinates of currently selected points
    # DBSCAN in sklearn runs on CPU, so we move data there
    selected_xyz = gaussians._xyz[mask].detach().cpu().numpy()
    
    # 2. Run DBSCAN
    # eps: The maximum distance between two samples for one to be considered as in the neighborhood.
    # min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    # Note: 'eps' depends heavily on your scene scale. 0.1 is usually good for normalized scenes.
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(selected_xyz)
    labels = clustering.labels_
    
    # 3. Find Largest Cluster
    unique_labels = set(labels)
    # Remove noise label (-1) if it exists
    if -1 in unique_labels:
        unique_labels.remove(-1)
    
    # If no valid clusters found (everything is noise), return empty mask
    if len(unique_labels) == 0:
        return torch.zeros_like(mask)
    
    # Find the label with the most points
    largest_label = max(unique_labels, key=lambda x: np.sum(labels == x))
    
    # 4. Create the Subset Mask
    # This boolean array corresponds specifically to the subset of points we passed to DBSCAN
    cluster_mask_subset = (labels == largest_label)
    
    # 5. Map back to Global Mask
    # We create a copy of the original mask, but we set the bits to False 
    # if the point wasn't part of the main cluster.
    refined_mask = mask.clone()
    
    # 'mask' acts as an index to the True values. We update ONLY those values 
    # with the result from our clustering.
    cluster_mask_subset_torch = torch.tensor(cluster_mask_subset, device=mask.device)
    refined_mask[mask] = cluster_mask_subset_torch
    
    return refined_mask

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
        model_name, pretrained=pretrained_source, device=device
    )
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
        "last_query": ""
    }

    # --- GUI LAYOUT ---
    with server.gui.add_folder("Semantic Editor"):
        # Search
        gui_query = server.gui.add_text("Target Object", initial_value="bicycle")
        
        gui_threshold = server.gui.add_slider(
            "Selection Threshold", 
            min=0.0, max=1.0, step=0.01, initial_value=0.22
        )

        # Cluster based selection
        gui_cluster_eps = server.gui.add_slider(
            "Cluster Spread (EPS)", 
            min=0.01, 
            max=1.0, 
            step=0.01, 
            initial_value=0.15,
            hint="Maximum distance between points to be considered neighbors."
        )

        btn_reset_all = server.gui.add_button("Reset All Edits", color="red")
        
        # Paint / Delete
        gui_color_picker = server.gui.add_rgb("Paint Color", initial_value=(1.0, 0.0, 0.0))
        btn_paint = server.gui.add_button("Apply Paint")
        btn_delete = server.gui.add_button("Delete Selection")
        
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

    # --- CORE RENDER LOOP ---
    def render_scene(highlight_mask=None, similarity_scores=None):
        """Fetches Gaussian state, culls invisible points, sends to Viser."""
        
        # 1. Get Opacities
        opacities = gaussians.get_opacity
        
        # 2. Smart Culling: Only render points with Opacity > 0.05
        # This fixes "ghosting" artifacts and sorting errors.
        vis_mask = (opacities.squeeze() > 0.05).cpu().numpy()
        
        if not np.any(vis_mask):
            return 

        # 3. Extract & Filter Geometry (CPU Transfer)
        means = gaussians.get_xyz.detach().cpu().numpy()[vis_mask]
        opacities_vis = opacities.detach().cpu().numpy()[vis_mask]
        scales = gaussians.get_scaling.detach().cpu().numpy()[vis_mask]
        quats = gaussians.get_rotation.detach().cpu().numpy()[vis_mask]
        
        # 4. Compute Covariances for Rendering
        Rs = tf.SO3(quats).as_matrix()
        covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)

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
        
        rgbs_vis = final_rgbs_gpu.cpu().numpy()[vis_mask]

        # 6. Send to Viser
        server.scene.add_gaussian_splats(
            "/scene",
            centers=means,
            rgbs=rgbs_vis,
            opacities=opacities_vis,
            covariances=covariances
        )

    # --- CALLBACKS ---
    def update_selection(_):
        query_text = gui_query.value
        threshold = gui_threshold.value
        state["last_query"] = query_text

        if not query_text:
            render_scene(highlight_mask=None, similarity_scores=None)
            gui_status.value = "Showing Original"
            return

        gui_status.value = f"Processing '{query_text}'..."
        
        with torch.no_grad():
            # Embed Text (CLIP)
            text_tokens = tokenizer([query_text]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            
            # Cast to FP16 to match Gaussian Features
            text_features = text_features.to(gaussian_features.dtype)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Similarity
            similarity = (gaussian_features @ text_features.T).squeeze()
            similarity = torch.nan_to_num(similarity, nan=0.0)
            
            # 2. Initial Raw Mask (Global Search)
            raw_mask = (similarity > threshold)
            
            # 3. Apply DBSCAN Refinement
            # Only run if we actually have points to cluster
            if raw_mask.sum() > 0:
                gui_status.value = "Refining Cluster..."
                state["current_mask"] = refine_mask_dbscan(
                    gaussians, 
                    raw_mask, 
                    eps=gui_cluster_eps.value
                )
            else:
                state["current_mask"] = raw_mask
            
            # 4. Render
            count = state["current_mask"].sum().item()
            gui_status.value = f"Selected {count} points (Max Sim: {similarity.max():.4f})"
            
            render_scene(highlight_mask=state["current_mask"], similarity_scores=similarity)
            torch.cuda.empty_cache()

    def handle_rotate(_):
        if state["current_mask"] is None: return
        angles = [slider_roll.value, slider_pitch.value, slider_yaw.value]
        print(f"Rotating selection by {angles} radians...")
        rotate_selection(gaussians, state["current_mask"], angles)
        
        # Reset sliders to 0.0 (Relative Transform)
        slider_roll.value = 0.0
        slider_pitch.value = 0.0
        slider_yaw.value = 0.0
        update_selection(None)

    def handle_scale(_):
        if state["current_mask"] is None: return
        factor = slider_scale.value
        print(f"Scaling selection by {factor}x...")
        scale_selection(gaussians, state["current_mask"], factor)
        
        # Reset slider to 1.0 (Relative Transform)
        slider_scale.value = 1.0
        update_selection(None)

    def handle_delete(_):
        if state["current_mask"] is None: return
        delete_selection(gaussians, state["current_mask"])
        update_selection(None)

    def handle_paint(_):
        if state["current_mask"] is None: return
        color_selection(gaussians, state["current_mask"], np.array(gui_color_picker.value))
        update_selection(None)
    
    def handle_move(_):
        if state["current_mask"] is None: return
        offset = [slider_x.value, slider_y.value, slider_z.value]
        move_selection(gaussians, state["current_mask"], offset)
        
        slider_x.value = 0.0
        slider_y.value = 0.0
        slider_z.value = 0.0
        update_selection(None)

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
        gui_query.value = ""
        gui_status.value = "Scene Reset"
        render_scene(highlight_mask=None, similarity_scores=None)
        print("Scene fully restored.")

    # --- BINDING ---
    gui_query.on_update(update_selection)
    gui_threshold.on_update(update_selection)
    
    btn_delete.on_click(handle_delete)
    btn_paint.on_click(handle_paint)
    btn_reset_all.on_click(handle_reset_scene)
    
    btn_move.on_click(handle_move)
    btn_rotate.on_click(handle_rotate)
    btn_scale.on_click(handle_scale)

    gui_cluster_eps.on_update(update_selection) # binding for cluster

    # Initial Render
    print("Pre-loading scene...")
    render_scene()
    
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