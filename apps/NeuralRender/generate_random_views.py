import argparse
from input_representation import mesh
from renderer.cameras import Camera
from renderer.rasterizer import Rasterizer
from renderer.lights import Lights
from renderer.shaders.basic_shader import Shader
from pytorch3d.renderer import MeshRenderer
from PIL import Image
import numpy as np
import os
import math
import random
import json

def sample_spherical(npoints: int, ndim:int=3):
    vec = np.random.randn(npoints,ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def cartesian_to_spherical(xyz: np.ndarray):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    dist = np.sqrt(xy + xyz[:, 2]**2)
    elev = np.arctan2(xyz[:, 2],  np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azim = np.arctan2(xyz[:, 1],  xyz[:, 0])
    elev = [math.degrees(x) for x in elev]
    azim = [math.degrees(x) for x in azim]
    return dist, elev, azim


def get_default_colors():
    a = 0.5
    ambient_color = ((a,a,a),)

    d = 0.3
    diffuse_color = ((d,d,d),)

    s = 0.2
    specular_color = ((s,s,s),)

    return ambient_color, diffuse_color, specular_color


def randomly_sample_colors():
    r = random.uniform(0.0,0.5) ; g = random.uniform(0.0,0.5) ; b = random.uniform(0.0,0.5)
    ambient_color = ((r,g,b),)

    r = random.uniform(0.0,0.5) ; g = random.uniform(0.0,0.5) ; b = random.uniform(0.0,0.5)
    diffuse_color = ((r,g,b),)

    r = random.uniform(0,0.3) ; g = random.uniform(0,0.3) ; b = random.uniform(0,0.3)
    specular_color = ((r,g,b),)

    return ambient_color,diffuse_color,specular_color

def randomly_sample_direction():
    xdir = random.uniform(0,1) ; ydir = random.uniform(0,1) ; zdir = random.uniform(0,1)
    direction = ((xdir,ydir,zdir),)

    return direction

def main(args):
    mesh_path = args.mesh_path
    mesh_instance = mesh.TriangleMesh(mesh_path=mesh_path)
    mesh_instance.load_pytorch_mesh_from_file()
    mesh_instance.load_trimesh_from_file()
    mesh_instance = mesh_instance.unit_normalize()
    bb_diagonal_length = mesh_instance.get_bb_diagonal_length()

    camera_cartesian_points = sample_spherical(args.num_views)
    dist, elev, azim = cartesian_to_spherical(camera_cartesian_points)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    params_dict = {}
    light_position = randomly_sample_direction()
    for i in range(args.num_views):
        print(i)
        camera_params_dict = {}
        camera_instance = Camera()
        camera_instance.lookAt(bb_diagonal_length, elev[i], azim[i])
        camera_params_dict['dist'] = float(bb_diagonal_length)
        camera_params_dict['elev'] = math.radians(elev[i])
        camera_params_dict['azim'] = math.radians(azim[i])
        camera_location = camera_instance.getLocation()
        camera_location = camera_location.cpu().detach().numpy()
        camera_params_dict['x'] = camera_location[0][0]
        camera_params_dict['y'] = camera_location[0][1]
        camera_params_dict['z'] = camera_location[0][2]

        #ambient_color, diffuse_color, specular_color = randomly_sample_colors()
        ambient_color, diffuse_color, specular_color = get_default_colors()

        light_params_dict = {}
        light_instance = Lights(light_type="point", ambient_color=ambient_color, diffuse_color=diffuse_color,
                                specular_color=specular_color)
        light_instance.setup_light(position=light_position)
        light_params_dict['ambient'] = list(ambient_color[0])
        light_params_dict['diffuse'] = list(diffuse_color[0])
        light_params_dict['specular'] = list(specular_color[0])
        light_params_dict['position'] = list(light_position[0])
        params_dict[i] = {}
        params_dict[i]['camera'] = camera_params_dict
        params_dict[i]['light'] = light_params_dict

        rasterizer_instance = Rasterizer()
        rasterizer_instance.init_rasterizer(camera_instance.camera,image_size=512)

        shader_instance = Shader()
        shader_instance.setup_shader(camera_instance.camera, light_instance.light)

        renderer_instance = MeshRenderer(rasterizer=rasterizer_instance.rasterizer, shader=shader_instance.shader)
        images = renderer_instance(mesh_instance.pytorch_mesh)

        np_image = images[0].cpu().detach().numpy()*255.0
        np_image[:,:,3] = 255.0
        np_image = np_image.astype('uint8')
        pil_image = Image.fromarray(np_image)
        pil_image.save(os.path.join(args.out_dir,str(i)+'.png'))

    with open(os.path.join(args.out_dir,"params.json"),'w') as fh:
        params_json =  json.dump(params_dict,fh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", help="Path to the mesh file")
    parser.add_argument("--num-views", type=int, help="Number of views")
    parser.add_argument("--out-dir", help="Path to the output directory")

    args = parser.parse_args()
    main(args)
