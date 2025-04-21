# code from trellis should modify


import os
import copy
import sys
import importlib
import argparse
# import pandas as pd
# from easydict import EasyDict as edict
from functools import partial
import numpy as np
import open3d as o3d
import utils3d


def voxelize(filepath):  # glb单位是m，ply单位是mm，需要归一化（trellis代码缺少归一化机制）
    mesh = o3d.io.read_triangle_mesh(filepath) #(os.path.join(output_dir, 'renders', scene_name, 'mesh.ply'))  # TriangleMesh with 15224 points and 17387 triangles.
    print(mesh)

    # normalization
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent().max()
    mesh.translate(-center)  # 平移中心到原点
    mesh.scale(1.0 / extent, center=(0, 0, 0))  # 归一化到 [-0.5, 0.5] 盒子内部

    # check vertices range (should be in the range [-0.5, 0.5])
    vertices = np.asarray(mesh.vertices)
    print("vertices range after normalization:", vertices.min(axis=0), vertices.max(axis=0))

    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)  #(15224, 3)
    print(vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices) # shape=(15224, 3)
    print(mesh.vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)) # VoxelGrid with 4096 voxels.
    print(voxel_grid)
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()]) # (4096, 3)
    print(vertices.shape)
    # breakpoint()
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    # vertices = (vertices + 0.5) / 64 - 0.5 # (4096, 3)
    
    utils3d.io.write_ply(os.path.join('assets', f'hello.ply'), vertices)  # if 3d, expected to be N,C,D,H,W
    # breakpoint()
    return {'voxelized': True, 'num_voxels': len(vertices)}


# voxelize("assets/scene_com.glb")
