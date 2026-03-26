import torch
import numpy as np
import faiss
import os
import open_clip
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
import torchvision
import math

# --- GAUSSIAN SPLATTING IMPORTS ---
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.inpainting_utils import InpaintingConfig, RealTimeGaussianInpainter
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

# --- BERT IMPORTS ---
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from peft import PeftModel

# =========================================================================================
#  1. PURE MATH HELPERS (No Viser Dependency)
# =========================================================================================

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
        # sample density points
        t = torch.linspace(0, 1, density, device=device).unsqueeze(1)
        segment = p0 + t * (p1 - p0)
        lines_xyz.append(segment)
    
    new_xyz = torch.cat(lines_xyz, dim=0)
    n_points = new_xyz.shape[0]

    C0 = 0.28209479177387814
    color_tensor = torch.tensor(color, device=device).float()
    features_dc = ((color_tensor - 0.5) / C0).unsqueeze(0).expand(n_points, 3).unsqueeze(1) # [N, 1, 3]
    features_rest = torch.zeros((n_points, gaussians._features_rest.shape[1], 3), device=device)
    
    diag = torch.norm(box_max - box_min)
    scale = math.log(max(diag.item() * 0.005, 1e-4)) # Thin visible lines
    scaling = torch.full((n_points, 3), scale, device=device)

    rotation = torch.zeros((n_points, 4), device=device)
    rotation[:, 0] = 1.0 # Identity q=[1,0,0,0]

    # inverse sigmoid of 0.99 for opacity
    op = math.log(0.99 / (1.0 - 0.99))
    opacity = torch.full((n_points, 1), op, device=device)

    def append_tensor(base, new_t):
        if base is None: return None
        return torch.nn.Parameter(torch.cat([base.detach(), new_t], dim=0).requires_grad_(True))

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

def get_rotation_matrix(r, p, y):
    """ Creates a 3x3 rotation matrix from Roll, Pitch, Yaw (radians) using PyTorch """
    # Rotation about X (Roll)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r), math.cos(r)]
    ])
    # Rotation about Y (Pitch)
    Ry = torch.tensor([
        [math.cos(p), 0, math.sin(p)],
        [0, 1, 0],
        [-math.sin(p), 0, math.cos(p)]
    ])
    # Rotation about Z (Yaw)
    Rz = torch.tensor([
        [math.cos(y), -math.sin(y), 0],
        [math.sin(y), math.cos(y), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx

def rpy_to_quat(r, p, y):
    """ Converts RPY to Quaternion (w, x, y, z) """
    cy = math.cos(y * 0.5)
    sy = math.sin(y * 0.5)
    cp = math.cos(p * 0.5)
    sp = math.sin(p * 0.5)
    cr = math.cos(r * 0.5)
    sr = math.sin(r * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([w, x, y, z])

def quat_mult(q1, q2):
    """ Hamilton product for batched quaternions """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

# =========================================================================================
#  2. GEOMETRY ENGINE (The "Hands")
# =========================================================================================

def rotate_selection(gaussians, mask, roll_pitch_yaw):
    print(f"   -> Action: Rotating selection by {roll_pitch_yaw} radians")
    device = gaussians._xyz.device
    r, p, y = roll_pitch_yaw
    
    with torch.no_grad():
        # 1. Rotate Positions
        center = gaussians._xyz[mask].mean(dim=0)
        R = get_rotation_matrix(r, p, y).to(device).float()
        
        xyz_centered = gaussians._xyz[mask] - center
        gaussians._xyz[mask] = (xyz_centered @ R.T) + center
        
        # 2. Rotate Orientations
        q_rot = rpy_to_quat(r, p, y).to(device).float()
        gaussians._rotation[mask] = quat_mult(q_rot, gaussians._rotation[mask])
        gaussians._rotation[mask] = torch.nn.functional.normalize(gaussians._rotation[mask], dim=-1)

def move_selection(gaussians, mask, offset_vector):
    print(f"   -> Action: Moving selection by {offset_vector}")
    with torch.no_grad():
        offset = torch.tensor(offset_vector, device=gaussians._xyz.device).float()
        gaussians._xyz[mask] += offset

def delete_selection(gaussians, mask):
    print(f"   -> Action: Deleting selection")
    with torch.no_grad():
        gaussians._opacity[mask] = -100.0

# =========================================================================================
#  3. BERT PARSER (The "Brain")
# =========================================================================================

class CommandInterpreter:
    def __init__(self, model_path):
        self.label_list = [
            "O", "B-TARGET", "I-TARGET", "B-ACTION", "I-ACTION",
            "B-DIRECTION", "I-DIRECTION", "B-ORIENTATION", "I-ORIENTATION",
            "B-ATTRIBUTE", "I-ATTRIBUTE"
        ]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for l, i in self.label2id.items()}
        
        base_model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        
        base_model = AutoModelForTokenClassification.from_pretrained(
            base_model_name, num_labels=len(self.label_list),
            id2label=self.id2label, label2id=self.label2id
        )
        
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
            if label == "TARGET": target_parts.append(word)
            elif label == "ATTRIBUTE": target_parts.insert(0, word)
            else: parsed[label] = word.lower()
        
        if target_parts: parsed['FULL_TARGET'] = " ".join(target_parts)
        return parsed

# =========================================================================================
#  4. MAIN RENDER LOGIC
# =========================================================================================

def render_set_custom(model_path, name, views, gaussians, pipeline, background, label, args):
    """ Modified version of your render function that saves to an 'edited' folder """
    render_path = os.path.join(model_path, name, f"renders_edited_{label}")
    makedirs(render_path, exist_ok=True)
    
    print(f"   -> Rendering {len(views)} frames to: {render_path}")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # FIXED: Added 'args' as the last argument
        output = render(view, gaussians, pipeline, background, args)
        rendering = output["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, f"{idx:05d}.png"))
    
def main(args, dataset_args, pipeline_args):
    device = "cuda"
    
    # --- 1. Load BERT ---
    print("\n[1/5] Loading BERT Logic...")
    bert_path = "./lora/token_classifier_lora_model"
    if not os.path.exists(bert_path):
        print(f"Error: BERT model not found at {bert_path}")
        return
    commander = CommandInterpreter(bert_path)

    # --- 2. Load CLIP ---
    print("[2/5] Loading CLIP Selection...")
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16")

    # --- 3. Load Scene ---
    print("[3/5] Loading Gaussian Scene...")
    gaussians = GaussianModel(dataset_args.sh_degree)
    scene = Scene(dataset_args, gaussians, shuffle=False)
    
    checkpoint = os.path.join(args.model_path, 'chkpnt0.pth')
    (model_params, _) = torch.load(checkpoint)
    gaussians.restore(model_params, args, mode='test')
    
    bg_color = [1,1,1] if dataset_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # --- 4. Load Features & Execute Edit ---
    print("[4/5] Processing Edit Command...")
    
    # 4a. Decode Features
    index = faiss.read_index(args.pq_index)
    inpainter = RealTimeGaussianInpainter(
        gaussians=gaussians,
        pq_index=index,
        config=InpaintingConfig(
            max_semantic_clusters=args.inpaint_k,
            blend_knn_k=args.inpaint_blend_k,
        ),
    )
    feat_idx = gaussians._language_feature.clone()
    valid_mask = (torch.sum(feat_idx, 1) != 255 * (index.coarse_code_size() + index.code_size))
    decoded_cpu = np.zeros((feat_idx.shape[0], 512), dtype=np.float32)
    decoded_cpu[valid_mask.cpu().numpy()] = index.sa_decode(feat_idx[valid_mask.cpu().numpy()].cpu().numpy())
    gaussian_features = torch.tensor(decoded_cpu, device=device, dtype=torch.float16)
    gaussian_features /= (gaussian_features.norm(dim=-1, keepdim=True) + 1e-5)

    # 4b. Parse Prompt
    prompt = args.edit_prompt
    parsed = commander.parse(prompt)
    print(f"   -> Parsed: {parsed}")
    
    target_text = parsed.get("FULL_TARGET", parsed.get("TARGET", ""))
    if not target_text:
        print("   -> Error: Could not identify a target object.")
        return

    # 4c. Select Object
    text_tokens = tokenizer([target_text]).to(device)
    text_emb = clip_model.encode_text(text_tokens).to(torch.float16)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    
    sims = (gaussian_features @ text_emb.T).squeeze()
    mask = (sims > args.threshold)
    print(f"   -> Selected {mask.sum().item()} gaussians for '{target_text}'")

    if mask.sum() > 0:
        # 4d. Apply Action
        action = parsed.get("ACTION", "").lower()
        direction = parsed.get("DIRECTION", "").lower()
        orientation = parsed.get("ORIENTATION", "").lower()

        if action in ["move", "slide", "push", "lift"]:
            offset = [0.0, 0.0, 0.0]
            dist = 1.0 
            if "left" in direction: offset[0] = -dist
            elif "right" in direction: offset[0] = dist
            elif "up" in direction: offset[1] = dist
            elif "down" in direction: offset[1] = -dist
            elif "forward" in direction: offset[2] = dist
            elif "back" in direction: offset[2] = -dist
            move_selection(gaussians, mask, offset)

        elif action in ["rotate", "turn", "spin"]:
            rads = 1.57 # 90 deg
            rpy = [0.0, 0.0, 0.0]
            if "counter" in orientation: rpy[1] = rads
            else: rpy[1] = -rads
            rotate_selection(gaussians, mask, rpy)

        elif action in ["delete", "remove", "drop"]:
            # Capture the exact bounding box of the removed item before any modifications.
            if args.draw_search_box and mask.sum() > 0:
                removed_xyz_for_box = gaussians._xyz[mask]
                custom_box_min = removed_xyz_for_box.min(dim=0).values.clone()
                custom_box_max = removed_xyz_for_box.max(dim=0).values.clone()
                diag_for_box = torch.norm(custom_box_max - custom_box_min).item()
                pad_for_box = max(0.01 * diag_for_box, 0.05) 
                custom_box_min -= pad_for_box
                custom_box_max += pad_for_box

            delete_selection(gaussians, mask)
            # Real-time 3D semantic patch inpainting executes immediately after deletion.
            stats = inpainter.inpaint(mask)
            if stats.get("status", 0) == 1:
                print(
                    f"   -> Inpainted hole with {stats['cloned']} cloned gaussians "
                    f"(boundary={stats['boundary']}, removed={stats['removed']})"
                )
            else:
                print(f"   -> Inpainting skipped/fallback (reason={stats.get('reason', -1)})")

            if args.draw_search_box and mask.sum() > 0:
                print("   -> Drawing search space bounding box in RED...")
                # Best effort to use the exact scaled bounds from inpainter, or default to the padded target.
                final_min = stats.get("box_min", custom_box_min)
                final_max = stats.get("box_max", custom_box_max)
                if final_min.numel() > 0:
                    draw_bounding_box_gaussians(gaussians, final_min, final_max, color=(1.0, 0.0, 0.0))
    else:
        print("   -> Warning: No objects selected. Rendering unchanged scene.")

    # --- 5. Render ---
    print(f"\n[5/5] Rendering Result...")
    safe_prompt_label = prompt.replace(" ", "_").lower()[:15]
    
    if not args.skip_train:
        # FIXED: Added 'args' at the end
        render_set_custom(dataset_args.model_path, "train", scene.getTrainCameras(), gaussians, pipeline_args, background, safe_prompt_label, args)

    if not args.skip_test:
        # FIXED: Added 'args' at the end
        render_set_custom(dataset_args.model_path, "test", scene.getTestCameras(), gaussians, pipeline_args, background, safe_prompt_label, args)

if __name__ == "__main__":
    parser = ArgumentParser(description="Edit & Render Script")
    model = ModelParams(parser, sentinel=True)
    pipe_params = PipelineParams(parser)
    
    # Custom Args
    parser.add_argument("--edit_prompt", type=str, required=True, help="Command to execute")
    parser.add_argument("--pq_index", type=str, required=True, help="Path to FAISS index")
    parser.add_argument("--threshold", type=float, default=0.2, help="CLIP Sensitivity")
    parser.add_argument("--inpaint_k", type=int, default=3, help="Semantic boundary clusters for inpainting")
    parser.add_argument("--inpaint_blend_k", type=int, default=5, help="KNN neighbors for seam blending")
    parser.add_argument("--draw_search_box", action="store_true", help="Draws a red bounding box around the inpainting search area")
    
    # Standard Render Args
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--include_feature", action="store_true", help="Include feature rendering")
    
    args = get_combined_args(parser)
    safe_state(args.quiet)
    
    main(args, model.extract(args), pipe_params.extract(args))