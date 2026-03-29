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

# --- NEW IMPORTS FOR BERT ---
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel

# =========================================================================================
#  MATH & GEOMETRY HELPERS 
# =========================================================================================

def rotate_selection(gaussians, mask, roll_pitch_yaw):
    """ Rotates the selection around its center of mass. """
    with torch.no_grad():
        center = gaussians._xyz[mask].mean(dim=0)
        rot_obj = tf.SO3.from_rpy_radians(*roll_pitch_yaw)
        R = torch.tensor(rot_obj.as_matrix(), device=gaussians._xyz.device, dtype=torch.float32)
        xyz_centered = gaussians._xyz[mask] - center
        gaussians._xyz[mask] = (xyz_centered @ R.T) + center
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

        gaussians._rotation[mask] = quat_mult(q_new, gaussians._rotation[mask])
        gaussians._rotation[mask] = torch.nn.functional.normalize(gaussians._rotation[mask], dim=-1)

def scale_selection(gaussians, mask, scale_factor):
    """ Scales the selection relative to its center of mass. """
    if scale_factor <= 0: return
    with torch.no_grad():
        center = gaussians._xyz[mask].mean(dim=0)
        gaussians._xyz[mask] = (gaussians._xyz[mask] - center) * scale_factor + center
        gaussians._scaling[mask] += np.log(scale_factor)

def delete_selection(gaussians, mask):
    """ 'Deletes' objects by setting opacity to -infinity. """
    with torch.no_grad():
        gaussians._opacity[mask] = -100.0

def color_selection(gaussians, mask, target_rgb):
    """ Paints selected objects by overwriting the 0th Spherical Harmonic. """
    SH_C0 = 0.28209479177387814
    target_sh = (target_rgb - 0.5) / SH_C0
    with torch.no_grad():
        new_color = torch.tensor(target_sh, device=gaussians._features_dc.device).float()
        gaussians._features_dc[mask, 0, :] = new_color

def move_selection(gaussians, mask, offset_vector):
    """ Translates selected objects in 3D space. """
    with torch.no_grad():
        offset = torch.tensor(offset_vector, device=gaussians._xyz.device).float()
        gaussians._xyz[mask] += offset

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
#  BERT COMMAND PARSER
# =========================================================================================
class CommandInterpreter:
    def __init__(self, model_path):
        print(f"[BERT] Loading LoRA model from {model_path}...")
        
        # EXACT LIST AND ORDER FROM YOUR NOTEBOOK (12 items)
        self.label_list = [
            "O",
            "B-ACTION",
            "B-TARGET",
            "B-ATTRIBUTE", "I-ATTRIBUTE",
            "B-DIRECTION",
            "B-ORIENTATION",
            "B-MAGNITUDE", "I-MAGNITUDE",
            "B-REFERENCE", "I-REFERENCE",
            "B-AXIS"
        ]
        
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        
        base_model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        # Load Base Model with 12 labels
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name, 
            num_labels=len(self.label_list),
            id2label=self.id2label, 
            label2id=self.label2id
        )
        
        # Load LoRA
        self.model = PeftModel.from_pretrained(base_model, model_path)
        self.model.eval()
        
        self.nlp = pipeline(
            "token-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            aggregation_strategy="simple", 
            device=0 if torch.cuda.is_available() else -1
        )

    def parse(self, text):
        results = self.nlp(text)
        parsed = {}
        target_parts = []
        
        # Keywords to help if the model is uncertain (Added scale words)
        action_keywords = ["move", "rotate", "scale", "delete", "paint", "color", "lift", "remove", "grow", "shrink", "resize", "bigger", "smaller"]

        for entity in results:
            label = entity['entity_group']
            word = entity['word'].strip().lower()
            
            # Map labels to logic
            if label == "TARGET":
                target_parts.append(word)
            elif label == "ATTRIBUTE":
                target_parts.insert(0, word)
            elif label == "ACTION":
                parsed['ACTION'] = word
            else:
                # Capture DIRECTION, ORIENTATION, MAGNITUDE, AXIS, etc.
                parsed[label] = word
        
        # Safety Check: If ACTION wasn't caught, force it from keywords.
        text_lower = text.lower().split()
        if 'ACTION' not in parsed:
            for word in text_lower:
                if word in action_keywords:
                    parsed['ACTION'] = word
                    break

        if target_parts:
            # Clean target parts of any stray action words or tiny fragments
            clean_target = [w for w in target_parts if w not in action_keywords and len(w) > 1]
            parsed['FULL_TARGET'] = " ".join(clean_target)
            
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
    
    # --- 2. DECODE LANGUAGE FEATURES ---
    print("[4/6] Loading FAISS Index and Decoding Features...")
    index = faiss.read_index(args.pq_index)
    language_features_idx = gaussians._language_feature.clone()
    check_valid = torch.sum(language_features_idx, 1)
    invalid_index = check_valid == 255 * (index.coarse_code_size() + index.code_size)
    decoded_features_cpu = np.zeros((language_features_idx.shape[0], 512), dtype=np.float32)
    valid_mask_cpu = (invalid_index.cpu() == False).numpy()
    decoded_features_cpu[valid_mask_cpu] = index.sa_decode(language_features_idx[valid_mask_cpu].cpu().numpy())
    gaussian_features = torch.tensor(decoded_features_cpu, device=device, dtype=torch.float16)
    del decoded_features_cpu
    torch.cuda.empty_cache()
    norm = gaussian_features.norm(dim=-1, keepdim=True)
    gaussian_features.div_(norm + 1e-5)

    # --- 3. START VISER SERVER ---
    print("[5/6] Starting Viser Server...")
    server = viser.ViserServer(port=args.port)
    
    state = { "current_mask": None, "last_query": "" }

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
            text_features = text_features.to(gaussian_features.dtype)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (gaussian_features @ text_features.T).squeeze()
            similarity = torch.nan_to_num(similarity, nan=0.0)
            
            state["current_mask"] = (similarity > threshold)
            max_sim = similarity.max().item()
            gui_status.value = f"Max Score: {max_sim:.4f}"
            
            render_scene(highlight_mask=state["current_mask"], similarity_scores=similarity)

    # --- FULLY UPGRADED AI COMMAND HANDLER ---
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
        text_lower = text.lower() # Raw text fallback for edge cases
        
        mask = state["current_mask"]
        
        # --- A. MOVE ---
        if action in ["move", "slide", "push", "lift"]:
            offset = [0.0, 0.0, 0.0]
            dist = 1.0 # Default distance
            # Check BERT direction tag or raw text as fallback
            if "left" in direction or "left" in text_lower: offset[0] = -dist
            elif "right" in direction or "right" in text_lower: offset[0] = dist
            elif "up" in direction or "up" in text_lower: offset[1] = dist
            elif "down" in direction or "down" in text_lower: offset[1] = -dist
            elif "forward" in direction or "forward" in text_lower: offset[2] = dist
            elif "back" in direction or "back" in text_lower: offset[2] = -dist
            
            move_selection(gaussians, mask, offset)
            gui_logs.value = f"Moved '{target_desc}'."

        # --- B. ROTATE ---
        elif action in ["rotate", "turn", "spin"]:
            rads = 1.57 # 90 degrees default
            rpy = [0.0, 0.0, 0.0]
            # Assume Y-axis (Up) rotation for now
            if "counter" in orientation or "counter" in text_lower: 
                rpy[1] = rads
            else: 
                rpy[1] = -rads # Clockwise by default
            
            rotate_selection(gaussians, mask, rpy)
            gui_logs.value = f"Rotated '{target_desc}'."
            
        # --- C. DELETE ---
        elif action in ["delete", "remove", "drop"]:
            delete_selection(gaussians, mask)
            gui_logs.value = f"Deleted '{target_desc}'."
            
        # --- D. COLOR / PAINT ---
        elif action in ["paint", "color"]:
            # Check the sentence for specific colors and map to RGB
            if "blue" in text_lower:
                rgb, c_name = [0.0, 0.0, 1.0], "Blue"
            elif "green" in text_lower:
                rgb, c_name = [0.0, 1.0, 0.0], "Green"
            elif "yellow" in text_lower:
                rgb, c_name = [1.0, 1.0, 0.0], "Yellow"
            elif "white" in text_lower:
                rgb, c_name = [1.0, 1.0, 1.0], "White"
            elif "black" in text_lower:
                rgb, c_name = [0.0, 0.0, 0.0], "Black"
            elif "purple" in text_lower:
                rgb, c_name = [0.5, 0.0, 0.5], "Purple"
            elif "orange" in text_lower:
                rgb, c_name = [1.0, 0.5, 0.0], "Orange"
            else:
                # Default fallback
                rgb, c_name = [1.0, 0.0, 0.0], "Red"
                
            color_selection(gaussians, mask, np.array(rgb))
            gui_logs.value = f"Painted '{target_desc}' ({c_name})."

        # --- E. SCALE ---
        elif action in ["scale", "resize", "grow", "shrink", "bigger", "smaller"]:
            scale_factor = 2.0 # Default to make it larger
            # Check if user wants it smaller
            if "shrink" in action or "smaller" in text_lower:
                scale_factor = 0.5
                
            scale_selection(gaussians, mask, scale_factor)
            gui_logs.value = f"Scaled '{target_desc}' (x{scale_factor})."
            
        else:
            gui_logs.value = f"Unknown action: '{action}'. Try move, rotate, scale, paint, or delete."

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
    args = get_combined_args(parser)
    
    if not os.path.exists(args.model_path):
        print(f"Error: The model path '{args.model_path}' does not exist.")
        exit(1)
        
    main(args, lp.extract(args), op.extract(args))