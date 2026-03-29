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
from utils.inpainting_utils import InpaintingConfig, RealTimeGaussianInpainter
import math

# --- NEW IMPORTS FOR BERT ---
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel

# =========================================================================================
#  MATH & GEOMETRY HELPERS (UNCHANGED)
# =========================================================================================

def rotate_selection(gaussians, mask, roll_pitch_yaw):
    """ Rotates the selection around its center of mass. """
    if mask is None or not mask.any(): return
    with torch.no_grad():
        center = gaussians._xyz[mask].mean(dim=0)
        rot_obj = tf.SO3.from_rpy_radians(*roll_pitch_yaw)
        R = torch.tensor(rot_obj.as_matrix(), device=gaussians._xyz.device, dtype=torch.float32)
        
        new_xyz = gaussians._xyz.clone()
        xyz_centered = new_xyz[mask] - center
        new_xyz[mask] = (xyz_centered @ R.T) + center
        gaussians._xyz.copy_(new_xyz)

        q_new = torch.tensor(rot_obj.wxyz, device=gaussians._rotation.device, dtype=torch.float32)        
        
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1.unbind(-1)
            w2, x2, y2, z2 = q2.unbind(-1)
            return torch.stack([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dim=-1)

        new_rot = gaussians._rotation.clone()
        new_rot[mask] = quat_mult(q_new, new_rot[mask])
        new_rot[mask] = torch.nn.functional.normalize(new_rot[mask], dim=-1)
        gaussians._rotation.copy_(new_rot)

def scale_selection(gaussians, mask, scale_factor):
    """ Scales the selection relative to its center of mass. """
    if scale_factor <= 0 or mask is None or not mask.any(): return
    with torch.no_grad():
        center = gaussians._xyz[mask].mean(dim=0)
        new_xyz = gaussians._xyz.clone()
        new_xyz[mask] = (new_xyz[mask] - center) * scale_factor + center
        gaussians._xyz.copy_(new_xyz)
        
        new_scale = gaussians._scaling.clone()
        new_scale[mask] += np.log(scale_factor)
        gaussians._scaling.copy_(new_scale)

def delete_selection(gaussians, mask):
    """ 'Deletes' objects by setting opacity to -infinity. """
    if mask is None or not mask.any(): return
    with torch.no_grad():
        new_op = gaussians._opacity.clone()
        new_op[mask] = -100.0
        gaussians._opacity.copy_(new_op)

def color_selection(gaussians, mask, target_rgb):
    """ Paints selected objects by overwriting the 0th Spherical Harmonic. """
    if mask is None or not mask.any(): return
    SH_C0 = 0.28209479177387814
    target_sh = (target_rgb - 0.5) / SH_C0
    with torch.no_grad():
        new_dc = gaussians._features_dc.clone()
        new_color = torch.tensor(target_sh, device=new_dc.device).float()
        new_dc[mask, 0, :] = new_color
        gaussians._features_dc.copy_(new_dc)

def move_selection(gaussians, mask, offset_vector):
    """ Translates selected objects in 3D space. """
    if mask is None or not mask.any(): return
    with torch.no_grad():
        offset = torch.tensor(offset_vector, device=gaussians._xyz.device).float()
        # Non-in-place modification to avoid leaf node issues
        new_xyz = gaussians._xyz.clone()
        new_xyz[mask] += offset
        gaussians._xyz.copy_(new_xyz)

def draw_bounding_box_gaussians(gaussians, box_min, box_max, color=(1.0, 0.0, 0.0), density=100):
    """ Creates visible 3D Gaussians along the edges of a bounding box """
    device = gaussians._xyz.device
    corners = [
        [box_min[0], box_min[1], box_min[2]],
        [box_max[0], box_min[1], box_min[2]],
        [box_max[0], box_max[1], box_min[2]],
        [box_min[0], box_max[1], box_min[2]],
        [box_min[0], box_min[1], box_max[2]],
        [box_max[0], box_min[1], box_max[2]],
        [box_max[0], box_max[1], box_max[2]],
        [box_min[0], box_max[1], box_max[2]]
    ]
    edges = [
        (0,1), (1,2), (2,3), (3,0), # bottom face
        (4,5), (5,6), (6,7), (7,4), # top face
        (0,4), (1,5), (2,6), (3,7)  # vertical edges
    ]
    lines_xyz = []
    for (i, j) in edges:
        p0 = torch.tensor(corners[i], device=device)
        p1 = torch.tensor(corners[j], device=device)
        t = torch.linspace(0, 1, density, device=device).unsqueeze(1)
        segment = p0 + t * (p1 - p0)
        lines_xyz.append(segment)
    
    new_xyz = torch.cat(lines_xyz, dim=0)
    n_points = new_xyz.shape[0]

    C0 = 0.28209479177387814
    color_tensor = torch.tensor(color, device=device).float()
    features_dc = ((color_tensor - 0.5) / C0).unsqueeze(0).expand(n_points, 3).unsqueeze(1)
    features_rest = torch.zeros((n_points, gaussians._features_rest.shape[1], 3), device=device)
    
    diag = torch.norm(box_max - box_min)
    scale = math.log(max(diag.item() * 0.005, 1e-4)) # Thin visible lines
    scaling = torch.full((n_points, 3), scale, device=device)

    rotation = torch.zeros((n_points, 4), device=device)
    rotation[:, 0] = 1.0 # Identity q=[1,0,0,0]

    op = math.log(0.99 / (1.0 - 0.99))
    opacity = torch.full((n_points, 1), op, device=device)

    def append_tensor(base, new_t):
        if base is None: return None
        return torch.nn.Parameter(torch.cat([base.detach(), new_t], dim=0).requires_grad_(True))

    with torch.no_grad():
        gaussians._xyz = append_tensor(gaussians._xyz, new_xyz)
        gaussians._features_dc = append_tensor(gaussians._features_dc, features_dc)
        gaussians._features_rest = append_tensor(gaussians._features_rest, features_rest)
        gaussians._scaling = append_tensor(gaussians._scaling, scaling)
        gaussians._rotation = append_tensor(gaussians._rotation, rotation)
        gaussians._opacity = append_tensor(gaussians._opacity, opacity)
        if hasattr(gaussians, '_language_feature') and gaussians._language_feature is not None:
            lang_feat = torch.zeros(
                (n_points, gaussians._language_feature.shape[1]), 
                device=device, 
                dtype=gaussians._language_feature.dtype
            )
            gaussians._language_feature = append_tensor(gaussians._language_feature, lang_feat)

def get_colormap_safe(name):
    try:
        return matplotlib.colormaps[name]
    except AttributeError:
        return cm.get_cmap(name)

def apply_overlay_heatmap(similarities, base_rgbs, threshold=0.2, colormap_name='turbo'):
    """ Overlays heatmap on matches, keeps original color for background. """
    similarities = torch.nan_to_num(similarities, nan=0.0, posinf=1.0, neginf=0.0)
    max_sim = similarities.max()
    if max_sim <= threshold: return base_rgbs 
    norm_sim = similarities / (max_sim + 1e-8)
    norm_sim = torch.clamp(norm_sim, 0, 1)
    cmap = get_colormap_safe(colormap_name)
    heatmap_colors_cpu = cmap(norm_sim.cpu().numpy())[:, :3]
    heatmap_colors = torch.tensor(heatmap_colors_cpu, device=base_rgbs.device, dtype=torch.float32)
    mask = (similarities > threshold).float().unsqueeze(-1).to(base_rgbs.device)
    final_colors = heatmap_colors * mask + base_rgbs * (1 - mask)
    return final_colors

# =========================================================================================
#  NEW: BERT COMMAND PARSER
# =========================================================================================

class CommandInterpreter:
    def __init__(self, model_path):
        print(f"[BERT] Loading LoRA model from {model_path}...")
        self.label_list = [
            "O", "B-TARGET", "I-TARGET", "B-ACTION", "I-ACTION",
            "B-DIRECTION", "I-DIRECTION", "B-ORIENTATION", "I-ORIENTATION",
            "B-ATTRIBUTE", "I-ATTRIBUTE"
        ]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        
        base_model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load Base
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name, num_labels=len(self.label_list),
            id2label=self.id2label, label2id=self.label2id
        )
        
        # Load LoRA
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        self.nlp = pipeline(
            "token-classification", model=self.model, tokenizer=self.tokenizer,
            aggregation_strategy="simple", device=0 if torch.cuda.is_available() else -1
        )

    def parse(self, text):
        results = self.nlp(text)
        parsed = {}
        target_parts = []
        
        for entity in results:
            label = entity['entity_group']
            word = entity['word'].strip()
            
            if label == "TARGET":
                target_parts.append(word)
            elif label == "ATTRIBUTE":
                target_parts.insert(0, word) # Adjective before noun
            else:
                parsed[label] = word.lower()
        
        if target_parts:
            parsed['FULL_TARGET'] = " ".join(target_parts)
        return parsed

# =========================================================================================
#  MAIN APPLICATION
# =========================================================================================

def main(args, dataset_args, pipeline_args):
    # --- 1. SETUP & LOAD MODELS ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print(f"\n[1/6] Loading OpenCLIP Model (ViT-B-16)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    
    # --- LOAD BERT ---
    print(f"[2/6] Loading BERT Command Parser...")
    bert_path = "./lora/token_classifier_lora_model" # Ensure this folder exists
    commander = None
    if os.path.exists(bert_path):
        commander = CommandInterpreter(bert_path)
    else:
        print(f"WARNING: BERT path {bert_path} not found. NLP commands disabled.")

    print(f"[3/6] Loading Gaussian Model from {args.model_path}...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    checkpoint_path = os.path.join(args.model_path, "chkpnt0.pth") 
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.restore(model_params, args, mode='test')

    # --- BACKUP ORIGINAL STATE ---
    orig_features_dc = gaussians._features_dc.clone()
    orig_opacity = gaussians._opacity.clone()
    orig_xyz = gaussians._xyz.clone()
    orig_rotation = gaussians._rotation.clone()
    orig_scaling = gaussians._scaling.clone()
    
    def decode_all_features():
        language_features_idx = gaussians._language_feature.clone()
        check_valid = torch.sum(language_features_idx, 1)
        invalid_index = check_valid == 255 * (index.coarse_code_size() + index.code_size)
        decoded_features_cpu = np.zeros((language_features_idx.shape[0], 512), dtype=np.float32)
        valid_mask_cpu = (invalid_index.cpu() == False).numpy()
        decoded_features_cpu[valid_mask_cpu] = index.sa_decode(language_features_idx[valid_mask_cpu].cpu().numpy())
        g_feat = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float16)
        del decoded_features_cpu
        torch.cuda.empty_cache()
        norm = g_feat.norm(dim=-1, keepdim=True)
        g_feat.div_(norm + 1e-5)
        return g_feat

    # --- 2. DECODE LANGUAGE FEATURES ---
    print("[4/6] Loading FAISS Index and Decoding Features...")
    index = faiss.read_index(args.pq_index)
    state = { "current_mask": None, "last_query": "", "gaussian_features": decode_all_features() }
    
    # Initialize real-time inpainter
    inpainter = RealTimeGaussianInpainter(
        gaussians=gaussians,
        pq_index=index,
        config=InpaintingConfig(
            max_semantic_clusters=3,
            blend_knn_k=5,
        ),
    )

    # --- 3. START VISER SERVER ---
    print("[5/6] Starting Viser Server...")
    server = viser.ViserServer(port=args.port)

    # --- GUI LAYOUT ---
    with server.gui.add_folder("Semantic Editor"):
        
        # --- NEW: AI COMMAND CENTER ---
        with server.gui.add_folder("AI Command Center"):
            gui_chat = server.gui.add_text("Natural Language", initial_value="Rotate the red box clockwise")
            btn_exec_ai = server.gui.add_button("Execute AI Command", color="violet")
            gui_logs = server.gui.add_text("AI Logs", initial_value="Waiting...", disabled=True)

        # --- EXISTING MANUAL CONTROLS ---
        server.gui.add_markdown("---") # Separator
        server.gui.add_markdown("**Manual Controls**")
        
        gui_query = server.gui.add_text("Target Object", initial_value="bicycle")
        gui_threshold = server.gui.add_slider("Selection Threshold", min=0.0, max=1.0, step=0.01, initial_value=0.22)
        gui_draw_bbox = server.gui.add_checkbox("Draw Search BBox", initial_value=args.draw_search_box)
        btn_reset_all = server.gui.add_button("Reset All Edits", color="red")
        
        gui_color_picker = server.gui.add_rgb("Paint Color", initial_value=(1.0, 0.0, 0.0))
        btn_paint = server.gui.add_button("Apply Paint")
        btn_delete = server.gui.add_button("Delete Selection")
        
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
        opacities = gaussians.get_opacity
        vis_mask = (opacities.squeeze() > 0.05).cpu().numpy()
        if not np.any(vis_mask): return 

        means = gaussians.get_xyz.detach().cpu().numpy()[vis_mask]
        opacities_vis = opacities.detach().cpu().numpy()[vis_mask]
        scales = gaussians.get_scaling.detach().cpu().numpy()[vis_mask]
        quats = gaussians.get_rotation.detach().cpu().numpy()[vis_mask]
        
        Rs = tf.SO3(quats).as_matrix()
        covariances = np.einsum("nij,njk,nlk->nil", Rs, np.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs)

        base_rgbs_gpu = (gaussians.get_features[:, 0, :].detach() * 0.28209479177387814 + 0.5).clamp(0, 1)
        final_rgbs_gpu = base_rgbs_gpu
        if similarity_scores is not None and highlight_mask is not None:
             final_rgbs_gpu = apply_overlay_heatmap(similarity_scores, base_rgbs_gpu, threshold=gui_threshold.value)
        
        rgbs_vis = final_rgbs_gpu.cpu().numpy()[vis_mask]

        server.scene.add_gaussian_splats(
            "/scene", centers=means, rgbs=rgbs_vis, opacities=opacities_vis, covariances=covariances
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
            text_tokens = tokenizer([query_text]).to(device)
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features.to(state["gaussian_features"].dtype)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (state["gaussian_features"] @ text_features.T).squeeze()
            similarity = torch.nan_to_num(similarity, nan=0.0)
            
            state["current_mask"] = (similarity > threshold)
            max_sim = similarity.max().item()
            gui_status.value = f"Max Score: {max_sim:.4f}"
            
            render_scene(highlight_mask=state["current_mask"], similarity_scores=similarity)

    # --- NEW: AI COMMAND HANDLER ---
    def handle_ai_command(_):
        text = gui_chat.value
        if not text or not commander: 
            gui_logs.value = "Error: No text or BERT not loaded."
            return
        
        gui_logs.value = "Thinking..."
        
        # 1. Parse Logic (Brain)
        parsed = commander.parse(text)
        gui_logs.value = f"Parsed: {parsed}"
        print(f"Parsed Command: {parsed}")
        
        # 2. Identify Target (Eyes)
        target_desc = parsed.get("FULL_TARGET", parsed.get("TARGET", ""))
        if not target_desc:
            gui_logs.value = "Error: I didn't catch the object name."
            return

        # Update the visual selection based on what BERT found
        gui_query.value = target_desc 
        update_selection(None) # Triggers CLIP search
        
        if state["current_mask"] is None or state["current_mask"].sum() == 0:
            gui_logs.value = f"Found 0 objects matching '{target_desc}'."
            return
        
        # 3. Execute Action (Hands)
        action = parsed.get("ACTION", "").lower()
        direction = parsed.get("DIRECTION", "").lower()
        orientation = parsed.get("ORIENTATION", "").lower()
        
        mask = state["current_mask"]
        
        if action in ["move", "slide", "push", "lift"]:
            offset = [0.0, 0.0, 0.0]
            dist = 1.0 # Default distance
            if "left" in direction: offset[0] = -dist
            elif "right" in direction: offset[0] = dist
            elif "up" in direction: offset[1] = dist
            elif "down" in direction: offset[1] = -dist
            elif "forward" in direction: offset[2] = dist
            elif "back" in direction: offset[2] = -dist
            
            move_selection(gaussians, mask, offset)
            gui_logs.value = f"Moved '{target_desc}' {direction}."

        elif action in ["rotate", "turn", "spin"]:
            rads = 1.57 # 90 degrees
            rpy = [0.0, 0.0, 0.0]
            # Assume Y-axis (Up) rotation for now
            if "counter" in orientation: rpy[1] = rads
            else: rpy[1] = -rads # Clockwise
            
            rotate_selection(gaussians, mask, rpy)
            gui_logs.value = f"Rotated '{target_desc}' {orientation}."
            
        elif action in ["delete", "remove", "drop"]:
            delete_selection(gaussians, mask)
            
            # --- START INPAINTING MODIFICATION ---
            stats = inpainter.inpaint(mask)
            if stats.get("status", 0) == 1:
                gui_logs.value = f"Deleted '{target_desc}' & Inpainted ({stats['cloned']} cloned)."
                if gui_draw_bbox.value and "box_min" in stats:
                    draw_bounding_box_gaussians(gaussians, stats["box_min"], stats["box_max"], color=(1.0, 0.0, 0.0))
            else:
                gui_logs.value = f"Deleted '{target_desc}'. (Inpaint skipped, reason={stats.get('reason', -1)})"
            
            # After appending elements, we must re-decode features
            state["gaussian_features"] = decode_all_features()
            # The mask needs to be resized to the new Gaussian count, we just reset the selection.
            state["current_mask"] = None
            gui_query.value = ""
            # --- END INPAINTING MODIFICATION ---
            
        elif action in ["paint", "color"]:
            # Basic fallback for painting red if detected
            color_selection(gaussians, mask, np.array([1.0, 0.0, 0.0]))
            gui_logs.value = f"Painted '{target_desc}' (Default Red)."

        # Refresh Scene
        render_scene(highlight_mask=mask, similarity_scores=None)

    # --- MANUAL HANDLERS ---
    def handle_rotate(_):
        if state["current_mask"] is None: return
        rotate_selection(gaussians, state["current_mask"], [slider_roll.value, slider_pitch.value, slider_yaw.value])
        slider_roll.value = slider_pitch.value = slider_yaw.value = 0.0
        update_selection(None)

    def handle_scale(_):
        if state["current_mask"] is None: return
        scale_selection(gaussians, state["current_mask"], slider_scale.value)
        slider_scale.value = 1.0
        update_selection(None)

    def handle_delete(_):
        if state["current_mask"] is None: return
        delete_selection(gaussians, state["current_mask"])
        stats = inpainter.inpaint(state["current_mask"])
        if stats.get("status", 0) == 1:
            gui_logs.value = f"Manual Delete & Inpaint ({stats['cloned']} cloned)."
            if gui_draw_bbox.value and "box_min" in stats:
                draw_bounding_box_gaussians(gaussians, stats["box_min"], stats["box_max"], color=(1.0, 0.0, 0.0))
        else:
            gui_logs.value = f"Manual Delete. (Inpaint skipped, reason={stats.get('reason', -1)})"
        
        state["gaussian_features"] = decode_all_features()
        state["current_mask"] = None
        gui_query.value = ""
        update_selection(None)

    def handle_paint(_):
        if state["current_mask"] is None: return
        color_selection(gaussians, state["current_mask"], np.array(gui_color_picker.value))
        update_selection(None)
    
    def handle_move(_):
        if state["current_mask"] is None: return
        move_selection(gaussians, state["current_mask"], [slider_x.value, slider_y.value, slider_z.value])
        slider_x.value = slider_y.value = slider_z.value = 0.0
        update_selection(None)

    def handle_reset_scene(_):
        with torch.no_grad():
            gaussians._features_dc.copy_(orig_features_dc)
            gaussians._opacity.copy_(orig_opacity)
            gaussians._xyz.copy_(orig_xyz)
            gaussians._rotation.copy_(orig_rotation)
            gaussians._scaling.copy_(orig_scaling)
        state["current_mask"] = None
        gui_query.value = ""
        gui_status.value = "Scene Reset"
        gui_logs.value = "Reset."
        render_scene(highlight_mask=None, similarity_scores=None)

    # --- BINDING ---
    gui_query.on_update(update_selection)
    gui_threshold.on_update(update_selection)
    btn_exec_ai.on_click(handle_ai_command) # Bind AI button
    
    btn_delete.on_click(handle_delete)
    btn_paint.on_click(handle_paint)
    btn_reset_all.on_click(handle_reset_scene)
    btn_move.on_click(handle_move)
    btn_rotate.on_click(handle_rotate)
    btn_scale.on_click(handle_scale)

    print("Pre-loading scene...")
    render_scene()
    print(f"SUCCESS! Interactive Editor running on http://localhost:{args.port}")
    
    while True: time.sleep(1.0)

if __name__ == "__main__":
    parser = ArgumentParser(description="Interactive Semantic Editor")
    lp = ModelParams(parser)
    op = PipelineParams(parser)
    parser.add_argument("--query", type=str, default="bicycle")
    parser.add_argument("--threshold", type=float, default=0.22) 
    parser.add_argument("--pq_index", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--draw_search_box", action="store_true", help="Draws a red bounding box around the inpainting search area")
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))