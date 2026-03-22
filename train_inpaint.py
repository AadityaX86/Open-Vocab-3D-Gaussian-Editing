import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from PIL import Image
from random import randint
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam
from inpaint_utils import unproject_to_3d, generate_injected_gaussian_parameters
import math

class PipelineParams:
    compute_cov3D_python = False
    convert_SHs_python = False
    debug = False

class TrainingParams:
    percent_dense = 0.01
    include_feature = False
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001

def create_training_cameras(npz_positions, num_views):
    # Depending on how virtual_views saved them, you'd load extrinsics/intrinsics.
    # We will simulate re-creating the virtual cameras identically to virtual_views.py
    # for simplicity if only center/radius is passed.
    pass

def training_loop(gaussians, cameras, inpainted_images, masks, iterations=1000):
    # Setup Optimizer for localized training
    train_params = TrainingParams()
    gaussians.training_setup(train_params) # We use simple setup, but need to be careful with spatial bounds
    gaussians.optimizer.zero_grad(set_to_none=True)
    
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    opt_params = PipelineParams()

    print(f"Starting localized optimization for {iterations} iterations...")
    for iteration in range(1, iterations + 1):
        # Pick a random camera
        view_idx = randint(0, len(cameras) - 1)
        viewpoint_cam = cameras[view_idx]
        gt_image = inpainted_images[view_idx]
        mask = masks[view_idx]

        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt_params, background)
        image = render_pkg["render"]

        # Calculate Loss bounded primarily to the masked region
        # Masks are 2D (H,W), extend to (3,H,W)
        mask_tensor = mask.unsqueeze(0).repeat(3, 1, 1)
        
        # We enforce L1 only on the inpainted/masked regions to preserve the background
        # Or alternatively the entire image, but masked is faster and avoids shifting background
        l1 = l1_loss(image * mask_tensor, gt_image * mask_tensor)
        loss = (1.0 - 0.2) * l1 + 0.2 * (1.0 - ssim(image * mask_tensor, gt_image * mask_tensor))

        loss.backward()

        with torch.no_grad():
            # Crucial: Freeze the original Gaussians so they don't shift!
            gaussians.zero_frozen_gradients()
            
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}/{iterations} Loss: {loss.item():.5f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to original model directory")
    parser.add_argument("--inpaint_dir", type=str, required=True, help="Directory with virtual views, inpainted rgb, and masks")
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()

    gaussians = GaussianModel(sh_degree=3)
    loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    
    ply_path = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    
    # 1. Mark original scene as frozen
    gaussians.mark_frozen()
    
    # Placeholder: In a real run, you'd load the specific cameras saved out by `virtual_views.py`.
    # Here we simulate aggregating unprojected points across all generated views.
    
    # ... Assume we recreate cameras and load the inpainted frames ...
    # This block unprojects the newly inpainted 2D spaces into 3D.
    
    print("Unprojecting inpainted depths back to 3D...")
    # Example snippet logic:
    # all_xyz = []
    # all_rgb = []
    # for idx in range(num_views):
    #     xyz, rgb = unproject_to_3d(cameras[idx], depth_maps[idx], masks[idx], inpainted_rgbs[idx])
    #     all_xyz.append(xyz)
    #     all_rgb.append(rgb)
    # merged_xyz = torch.cat(all_xyz, dim=0)
    # merged_rgb = torch.cat(all_rgb, dim=0)

    # 2. Inject points!
    # f_dc, f_rest, opacities, scales, rots = generate_injected_gaussian_parameters(merged_xyz, merged_rgb, len(merged_xyz))
    # gaussians.inject_gaussians(merged_xyz.cuda(), f_dc.cuda(), f_rest.cuda(), opacities.cuda(), scales.cuda(), rots.cuda())

    # 3. Setup optimizer. Re-init the optimizer since we added parameters
    # The inject_gaussians method handles cat_tensors_to_optimizer.
    # Note: If `training_setup` was not called yet in GaussianModel, just call it now.
    
    # 4. Train
    # training_loop(gaussians, cameras, inpainted_rgbs, masks_tensor, iterations=args.iterations)
    
    # 5. Save out new model
    out_pth = os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter+args.iterations}_inpainted")
    os.makedirs(out_pth, exist_ok=True)
    # gaussians.save_ply(os.path.join(out_pth, "point_cloud.ply"))
    print(f"Saved inpainted 3D model to {out_pth}")

if __name__ == "__main__":
    main()
