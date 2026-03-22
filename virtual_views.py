import torch
import numpy as np
import os
import math
from PIL import Image
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.cameras import MiniCam

def get_virtual_cameras(center, radius, num_views=10, fov=60.0, height=512, width=512):
    cameras = []
    
    fov_rad = fov * math.pi / 180.0
    znear = 0.01
    zfar = 100.0
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_rad, fovY=fov_rad).transpose(0, 1).cuda()
    
    for i in range(num_views):
        angle = (i / num_views) * 2 * math.pi
        
        # Calculate camera position orbiting around the center
        cam_x = center[0] + radius * math.cos(angle)
        cam_y = center[1] # keep height same as center for now
        cam_z = center[2] + radius * math.sin(angle)
        
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Look at the center
        forward = center - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        # Up vector
        up = np.array([0, 1, 0])
        
        # Right vector
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)
        
        # Recompute up vector
        up = np.cross(forward, right)
        
        # Construct rotation matrix (camera to world)
        R = np.eye(3)
        R[:, 0] = right
        R[:, 1] = -up # Invert Y for graphics convention
        R[:, 2] = forward
        
        # World to camera rotation
        R = R.T
        T = -R @ cam_pos
        
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]
        
        cam = MiniCam(
            width=width,
            height=height,
            fovy=fov_rad,
            fovx=fov_rad,
            znear=znear,
            zfar=zfar,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform
        )
        cam.camera_center = camera_center
        cameras.append(cam)
        
    return cameras

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--center", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Target object center (X, Y, Z)")
    parser.add_argument("--radius", type=float, default=2.0, help="Distance of virtual cameras from the center")
    parser.add_argument("--num_views", type=int, default=10, help="Number of virtual views to generate")
    parser.add_argument("--out_dir", type=str, default="virtual_views", help="Output directory")
    args = parser.parse_args()

    # Create model and load checkpoint
    gaussians = GaussianModel(sh_degree=3)
    loaded_iter = searchForMaxIteration(os.path.join(args.model_path, "point_cloud"))
    print(f"Loading trained model at iteration {loaded_iter}")
    gaussians.load_ply(os.path.join(args.model_path, "point_cloud", f"iteration_{loaded_iter}", "point_cloud.ply"))

    # Generate virtual views
    center = np.array(args.center)
    cameras = get_virtual_cameras(center, radius=args.radius, num_views=args.num_views)

    # Output paths
    rgb_dir = os.path.join(args.out_dir, "rgb")
    depth_dir = os.path.join(args.out_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print(f"Rendering {len(cameras)} views...")
    for idx, cam in enumerate(cameras):
        # We pass a simple namespace or minimal args matching the pipeline properties
        class PipelineParams:
            compute_cov3D_python = False
            convert_SHs_python = False
            debug = False
        
        # Render
        render_pkg = render(cam, gaussians, PipelineParams(), background)
        
        # Save RGB
        render_rgb = render_pkg["render"].permute(1, 2, 0).cpu().numpy()
        render_rgb = (np.clip(render_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        Image.fromarray(render_rgb).save(os.path.join(rgb_dir, f"{idx:03d}.png"))
        
        # Save Depth
        # Normalize depth map for visualization/saving (for LaMa we actually need the 3D metric depth later)
        render_depth = render_pkg["depth"].squeeze(0).cpu().numpy()
        np.save(os.path.join(depth_dir, f"{idx:03d}.npy"), render_depth)
        
        # For visualization as 8-bit image
        min_d, max_d = np.percentile(render_depth, [1, 99])
        depth_vis = np.clip((render_depth - min_d) / (max_d - min_d + 1e-5), 0, 1)
        depth_vis = (depth_vis * 255).astype(np.uint8)
        Image.fromarray(depth_vis).save(os.path.join(depth_dir, f"{idx:03d}.png"))

    print("Virtual views generated successfully!")

if __name__ == "__main__":
    main()
