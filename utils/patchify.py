# here we patchify envmap to patches.
# use blender for multiview snapshot with camera params. (take care of view density while sampling) (then goes to blender_render.py)
# or `pip install patchify` (only for image, then adding projection matrix info)

# from patchify import patchify, unpatchify
# import cv2
import OpenEXR
import trimesh
import numpy as np
# from matplotlib import pyplot as plt
import torch
import voxelize

def patchify(x: torch.Tensor, patch_size: int):
    """
    Patchify a tensor.

    Args:
        x (torch.Tensor): (N, C, *spatial) tensor
        patch_size (int): Patch size
    """
    DIM = x.dim() - 2 # 这里是任意维度都通用的DIM的写法，减掉的是NC
    # print(x.dim(), DIM)
    for d in range(2, DIM + 2):
        assert x.shape[d] % patch_size == 0, f"Dimension {d} of input tensor must be divisible by patch size, got {x.shape[d]} and {patch_size}"

    x = x.reshape(*x.shape[:2], *sum([[x.shape[d] // patch_size, patch_size] for d in range(2, DIM + 2)], [])) # [1, 3, 32, 64, 64, 64]
    # print("step1:", x.shape)
    x = x.permute(0, 1, *([2 * i + 3 for i in range(DIM)] + [2 * i + 2 for i in range(DIM)])) # [1, 3, 64, 64, 32, 64]
    # np.save('debug/patchified_envmap.npy', x.detach().cpu().numpy())
    torch.save(x.detach().cpu(), 'debug/patchified_data.pt')
    # print("step2:", x.shape) 

    x = x.reshape(x.shape[0], x.shape[1] * (patch_size ** DIM), *(x.shape[-DIM:]))  # [1, 12288, 32, 64]), flattened
    # print("step3:", x.shape)
    return x

if __name__ == "__main__":

    pixel_list = []
    
    ## load exr file
    # with OpenEXR.File("assets/blue_photo_studio_4k.exr") as infile: # 2048 * 4096

    #     header = infile.header()
    #     print(f"type={header['type']}")
    #     print(f"compression={header['compression']}")

    #     RGB = infile.channels()["RGB"].pixels  # HWC, numpy.ndarray, (2048, 4096, 3), np.float32
    #     RGB_tensor = torch.tensor(RGB.tolist(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) # N,C,H,W

    ## load 3d scene
    scene = trimesh.load("assets/scene_com.glb")
    voxelize()
    
    
    # patchify(RGB_tensor, 64)  # 32 * 64
    patchify(scene, 32)  # 32 * 64