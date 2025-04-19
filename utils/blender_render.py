# code modified from https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/blender_script/render.py

import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob


"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

def init_render(engine='CYCLES', resolution=512, geo_mode=False, mask_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'BW' if mask_mode else 'RGBA'
    bpy.context.scene.render.film_transparent = True # 渲染出来背景是否透明
    
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.samples = 128 if not geo_mode else 512 # 1 too low, bad quality
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True

    if mask_mode:  # 可以优化分支
        bpy.context.scene.cycles.samples = 1
        bpy.context.scene.cycles.diffuse_bounces = 0
        bpy.context.scene.cycles.glossy_bounces = 0
        bpy.context.scene.cycles.transparent_max_bounces = 0
        bpy.context.scene.cycles.transmission_bounces = 0
        bpy.context.scene.cycles.use_denoising = False
        # bpy.context.scene.render.use_antialiasing = False # removed in 3.0
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
# def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False, envmap_path=None):
#     if not any([save_depth, save_normal, save_albedo, save_mist]) and envmap_path is None:
#         return {}, {}
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}
    outputs = {}
    spec_nodes = {}
    bpy.context.scene.use_nodes = True
    for view_layer in bpy.context.scene.view_layers:
        print(view_layer.name)
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['ViewLayer'].use_pass_mist = save_mist

    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)

    render_layers = nodes.new('CompositorNodeRLayers')
   
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = ''
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = 'PNG'
        depth_file_output.format.color_depth = '16'
        depth_file_output.format.color_mode = 'BW'
        # Remap to 0-1
        map = nodes.new(type="CompositorNodeMapRange")
        map.inputs[1].default_value = 0  # (min value you will be getting)
        map.inputs[2].default_value = 10 # (max value you will be getting)
        map.inputs[3].default_value = 0  # (min value you will map to)
        map.inputs[4].default_value = 1  # (max value you will map to)
        
        links.new(render_layers.outputs['Depth'], map.inputs[0])
        links.new(map.outputs[0], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        spec_nodes['depth_map'] = map
    print("save normals")
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        normal_file_output.format.color_mode = 'RGB'
        normal_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
    print("save albedos")
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
    print("save mist")    
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = ''
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
    
    return outputs, spec_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000 # 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000 # 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000 # 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light, 
    }

def init_environment_map(envmap_path):

    world = bpy.data.worlds['World']
    world.use_nodes = True

    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # 清空当前 World 节点
    nodes.clear()

    # 加载 HDR 贴图节点
    env_node = nodes.new('ShaderNodeTexEnvironment')
    try:
        env_node.image = bpy.data.images.load(envmap_path)
    except:
        print(f"❌ 无法加载 HDR 图像：{envmap_path}")
        return None
    env_node.location = (-300, 0)

    # 背景与输出节点
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.location = (0, 0)
    bg_node.inputs['Strength'].default_value = 2.0
    output_node = nodes.new('ShaderNodeOutputWorld')
    output_node.location = (300, 0)

    # 连接
    links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    return env_node


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)

def hide_all_objects() -> None:
    """Hides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_render = True

    # Hide all collections
    for col in bpy.data.collections:
        col.hide_render = True        


def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
     for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material(material_type='diffuse', mask_only=False):
    print("[INFO] start overriding material")
    # Clear existing materials
    new_mat = bpy.data.materials.new(name="Override_Test_Material")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')

    if material_type == 'diffuse':
        # 灰色漫反射
        print("diffuse")
        bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)  # 灰色
        bsdf.inputs[1].default_value = 1.0                 # 最大粗糙度
        new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    elif material_type == 'glossy':
        # 镜面反射
        print("glossy")
        glossy = new_mat.node_tree.nodes.new('ShaderNodeBsdfGlossy')
        glossy.inputs[0].default_value = (1.0, 1.0, 1.0, 1)  # 白色
        glossy.inputs[1].default_value = 0.0                 # 完全平滑
        new_mat.node_tree.links.new(glossy.outputs['BSDF'], output.inputs['Surface'])

    elif material_type == 'metal':
        # 金属表面
        print("metal")
        principled = new_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        principled.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1)  # 银灰色
        principled.inputs['Metallic'].default_value = 1.0
        principled.inputs['Roughness'].default_value = 0.2
        new_mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    elif material_type == 'glass':
        # 玻璃材质 - 使用 Principled BSDF
        print("glass (Principled)")
        glass = new_mat.node_tree.nodes.new('ShaderNodeBsdfPrincipled')
        # 设置基础颜色（无色）
        glass.inputs['Base Color'].default_value = (1.0, 1.0, 1.0, 1)  
        # 关键：设置完全透光
        glass.inputs['Transmission'].default_value = 1.0               
        # 折射率，普通玻璃 = 1.45
        glass.inputs['IOR'].default_value = 1.45                       
        # 粗糙度，0.0 = 完美平滑，0.2 = 有点磨砂
        glass.inputs['Roughness'].default_value = 0.2                  
        # 金属度保持0（非金属）
        glass.inputs['Metallic'].default_value = 0.0                   
        # 链接输出
        new_mat.node_tree.links.new(glass.outputs['BSDF'], output.inputs['Surface'])

    elif material_type == 'blue':
        # 蓝色漫反射
        print("blue")
        blue = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        blue.inputs[0].default_value = (0.0, 0.3, 1.0, 1)   # 蓝色
        blue.inputs[1].default_value = 0.5                  # 中等粗糙度
        new_mat.node_tree.links.new(blue.outputs['BSDF'], output.inputs['Surface'])
    
    elif material_type == 'warm':
        # 暖色漫反射
        print("warm")
        warm = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        warm.inputs[0].default_value = (1.0, 0.5, 0.3, 1)    # 橙色偏暖
        warm.inputs[1].default_value = 0.5                   # 中等粗糙度
        new_mat.node_tree.links.new(warm.outputs['BSDF'], output.inputs['Surface'])

    else:
        print("Unrecognized material. Should be diffuse / glossy / metal / glass / blue / warm.")
        return
    # bpy.context.scene.view_layers['View Layer'].material_override = new_mat
    bpy.context.view_layer.material_override = new_mat # blender 3.0.1

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset

def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    # Initialize context
    print("Initializing Blender context...")
    print("Initializing render ...")
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode, mask_mode=arg.mask_only)
    print("Initializing nodes ...")
    outputs, spec_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )
    # print("init nodes output:", outputs, spec_nodes)
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    elif arg.bg_only:
        hide_all_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()

    print('[INFO] Scene initialized.')

    init_lighting()
    init_environment_map(args.envmap_path)
    print('[INFO] Environment map initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    if arg.geo_mode or arg.mask_only: # 后面可以把geo/bg/mask合并成一个参数
        print("[INFO] geo mode started.")
        override_material(arg.material_type, arg.mask_only) # material_type 目前可以作为和diffuse等并列的一个选项，后面再优化这里的逻辑把material合并到geo里面去
        print("[INFO] material overrode.")
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }
    views = json.loads(arg.views)
    for i, view in enumerate(views):
        print("render start")
        cam.location = (
            view['radius'] * np.cos(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['yaw']) * np.cos(view['pitch']),
            view['radius'] * np.sin(view['pitch'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        if arg.save_depth:
            spec_nodes['depth_map'].inputs[1].default_value = view['radius'] - 0.5 * np.sqrt(3)
            spec_nodes['depth_map'].inputs[2].default_value = view['radius'] + 0.5 * np.sqrt(3)
        
        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            print(os.path.join(arg.output_folder, f'{i:03d}_{name}'))
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            print("output.file_slots[0].path", output.file_slots[0].path)
            
        # Render the scene
        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            print("render the scene")
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
            
        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            "transform_matrix": get_transform_matrix(cam)
        }
        if arg.save_depth:
            metadata['depth'] = {
                'min': view['radius'] - 0.5 * np.sqrt(3),
                'max': view['radius'] + 0.5 * np.sqrt(3)
            }
        to_export["frames"].append(metadata)
    
    # Save the camera parameters
    with open(os.path.join(arg.output_folder, 'transforms.json'), 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.export_mesh.ply(filepath=os.path.join(arg.output_folder, 'mesh.ply'))

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=str, help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--envmap_path', type=str, help='Path to environment map for lighting, prefer .HDR, .EXR for high dynamic range.')
    parser.add_argument('--geo_mode', action='store_true', help='Geometry mode for rendering.')
    parser.add_argument('--material_type', type=str, default='diffuse',
                        choices=['diffuse', 'glossy', 'metal', 'glass', 'blue', 'warm'],
                        help='Choose material type: diffuse / glossy / metal / glass / blue / warm.')
    parser.add_argument('--save_depth', action='store_true', help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', help='Save the albedo maps.')
    parser.add_argument('--save_mist', action='store_true', help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', help='Save the mesh as a .ply file.')
    parser.add_argument('--bg_only', action='store_true', help='Only render the background.')
    parser.add_argument('--mask_only', action='store_true', help='[!DO NOT USE THIS MODE! Simplest solution is to use alpha channel for masks.] Render object only into masks.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
    