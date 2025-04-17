# part of code modified from lss3d, hunyuan3d2 and trellis

import os
import torch
import json
import numpy as np
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from utils.lds import sphere_hammersley_sequence
from subprocess import DEVNULL, call
from PIL import Image
from math import exp
from lpips import LPIPS

BLENDER_LINK = 'https://download.blender.org/release/Blender3.0/blender-3.0.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/tmp'
BLENDER_PATH = f'{BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64/blender'

def install_blender():
    if not os.path.exists(BLENDER_PATH):
        os.system('sudo apt-get update')
        os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-3.0.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')

def render(file_path, output_type, output_dir, num_views):
    output_folder = os.path.join(output_dir, 'renders', output_type)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]

    # Setup 
    "setting up blender render ..."
    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'utils', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        # '--geo_mode',
        # '--material_type', 'diffuse',
        # '--save_depth',
        '--envmap_path', 'assets/blue_photo_studio_4k.exr',
        # '--save_mesh',  # comment out for bg only mode
        # '--bg_only'
        # '--mask_only'
        # '--depth_only'
    ] # 
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    # call(args, stdout=DEVNULL, stderr=DEVNULL)
    debug_file = "outputs/bpy_debug.txt"
    with open(debug_file, 'w') as f:
        call(args, stdout=f, stderr=DEVNULL)

    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'name': output_type, 'rendered': True}

def get_image(root, instance):
        with open(os.path.join(root, 'renders', instance, 'transforms.json')) as f:
            metadata = json.load(f)
        n_views = len(metadata['frames'])
        view = np.random.randint(n_views)
        metadata = metadata['frames'][view]
        fov = metadata['camera_angle_x']
        intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))
        c2w = torch.tensor(metadata['transform_matrix'])
        c2w[:3, 1:3] *= -1
        extrinsics = torch.inverse(c2w)

        image_path = os.path.join(root, 'renders', instance, metadata['file_path'])
        image = Image.open(image_path)
        image_size = image.size()[0] # default square shape
        alpha = image.getchannel(3)
        image = image.convert('RGB')
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        alpha = alpha.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        alpha = torch.tensor(np.array(alpha)).float() / 255.0
        
        return {
            'image': image,
            'alpha': alpha,
            'extrinsics': extrinsics,
            'intrinsics': intrinsics,
        }

def bake_from_multiview(views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

def psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    return 20 * torch.log10(max_val / torch.sqrt(mse))


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

loss_fn_vgg = None
def lpips(img1, img2, value_range=(0, 1)):
    global loss_fn_vgg
    if loss_fn_vgg is None:
        loss_fn_vgg = LPIPS(net='vgg').cuda().eval()
    # normalize to [-1, 1]
    img1 = (img1 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    img2 = (img2 - value_range[0]) / (value_range[1] - value_range[0]) * 2 - 1
    return loss_fn_vgg(img1, img2).mean()

# io 
print("read mesh in obj format")
obj_file = "/root/autodl-tmp/gaodongyu/MVRLT/assets/white_diffuse_ball.obj"
mesh = trimesh.load_mesh(obj_file)
vertices_tensor = torch.tensor(mesh.vertices, dtype=torch.float32)  # Shape: (num_vertices, 3)
faces_tensor = torch.tensor(mesh.faces, dtype=torch.int64)  # Shape: (num_faces, 3)
tex = TexturesVertex(verts_features=[torch.ones_like(vertices_tensor)])
mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor], textures=tex)

texture_dir = "/root/autodl-tmp/gaodongyu/MVRLT/outputs/renders/gt"
image_list = [Image.open(os.path.join(texture_dir, file_path)) for file_path in os.listdir(texture_dir)]  # distill more knowledge from self-supervised camparam/DINO/CLIP/flow(dense view)/depth/MVSNet/nerf/3dgs for richer representation


# render gt mv
print("define render function ...")
output_dir = "/root/autodl-tmp/gaodongyu/MVRLT/outputs"
num_views = 30
output_type = 'gt' #'textured', 'mask', 'depth', 'normal', ..
# render(obj_file, output_type, output_dir=output_dir, num_views=num_views) # wrap to another function and no need to call with commenting each time

# get cam params and bake
cameras_list = get_image()
texture, mask = bake_from_multiview()
mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
texture = texture_inpaint(texture, mask_np)
MeshRender.set_texture(texture)  # alias with blender render; Hunyuan render with rasters
textured_mesh = MeshRender.save_mesh() # alias with blender render

# render tex mv
output_type = 'results'
render(textured_mesh, output_type,  output_dir=output_dir, num_views=num_views) # ideally should combine render texmv - compare into optimization loop for grad propogation; should change blender render to differetiable mesh render.

# compare
MAE(gt, rendered)
MSE(gt, rendered)
PSNR(gt, rendered)
LPIPS(gt, rendered)
SSIM(gt, rendered)