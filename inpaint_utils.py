import torch
import numpy as np

def unproject_to_3d(camera, depth_map, mask_image, rgb_image):
    """
    Unprojects a 2D depth map to 3D points based on a camera's view matrix and mask.
    
    Args:
        camera: scene.cameras.MiniCam or Camera object.
        depth_map: (H, W) array of metric depth.
        mask_image: (H, W) array boolean/binary mask of the inpainted region.
        rgb_image: (H, W, 3) pixel colors.
        
    Returns:
        xyz: (N, 3) torch tensor of 3D point coordinates.
        colors: (N, 3) torch tensor of RGB colors.
    """
    H, W = depth_map.shape
    
    # Get active pixels in the mask
    ys, xs = np.nonzero(mask_image)
    if len(ys) == 0:
        return torch.empty((0, 3)), torch.empty((0, 3))
        
    depths = depth_map[ys, xs]
    colors = rgb_image[ys, xs] / 255.0  # Normalize to [0, 1]
    
    # Reconstruct Image-plane coordinates (Normalized Device Coordinates equivalent)
    # The rasterizer uses standard projection
    fovY = camera.FovY if hasattr(camera, 'FovY') else camera.fovy
    fovX = camera.FovX if hasattr(camera, 'FovX') else camera.fovx
    
    # Calculate focal lengths
    focal_y = H / (2.0 * np.tan(fovY / 2.0))
    focal_x = W / (2.0 * np.tan(fovX / 2.0))
    
    # Convert pixels to view-space coordinates
    cx = W / 2.0
    cy = H / 2.0
    
    # +Z is forward in view space
    X_view = (xs - cx) * depths / focal_x
    Y_view = (ys - cy) * depths / focal_y
    Z_view = depths
    
    # Construct view-space points (N, 4)
    view_points = np.stack([X_view, Y_view, Z_view, np.ones_like(Z_view)], axis=1)
    
    # Get view-to-world transform
    # camera.world_view_transform transforms world to view
    # Inverse transforms view to world
    c2w = torch.inverse(camera.world_view_transform).cpu().numpy()
    
    # Apply transform
    world_points = view_points @ c2w
    
    # Extract XYZ
    xyz = world_points[:, :3]
    
    return torch.tensor(xyz, dtype=torch.float32), torch.tensor(colors, dtype=torch.float32)

def generate_injected_gaussian_parameters(xyz, colors, num_points):
    """
    Generates initial Gaussian parameters based on unprojected positions and colors.
    """
    # RGB to SH conversion function logic
    # C0 = 0.28209479177387814
    # shs = (colors - 0.5) / C0 (Simplified initialization)
    f_dc = ((colors - 0.5) / 0.28209479177387814).unsqueeze(1).contiguous()
    
    # Rest SH features (zeros for initialization)
    f_rest = torch.zeros((num_points, 15, 3), dtype=torch.float32)
    
    # Small initial scaling
    # Assume target scene size, arbitrary small scale representing point size
    scales = torch.ones((num_points, 3), dtype=torch.float32) * -3.0 # log scale ~0.05
    
    # Initial rotations (identity quaternion: [1, 0, 0, 0])
    rots = torch.zeros((num_points, 4), dtype=torch.float32)
    rots[:, 0] = 1.0
    
    # High opacity to block backgrounds
    opacities = torch.ones((num_points, 1), dtype=torch.float32) * 5.0 # inverse sigmoid of approx 0.99
    
    return f_dc, f_rest, opacities, scales, rots

def run_lama_inpainting(rgb_path, mask_path, out_path):
    """
    Helper function to run simple_lama_inpainting if installed.
    You can pip install simple-lama-inpainting
    """
    try:
        from simple_lama_inpainting import SimpleLama
        from PIL import Image
        
        lama = SimpleLama()
        
        img = Image.open(rgb_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        result = lama(img, mask)
        result.save(out_path)
        print(f"Saved inpainted result to {out_path}")
        return result
    except ImportError:
        print("Install simple_lama_inpainting: pip install simple-lama-inpainting")
        return None
