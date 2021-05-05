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

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(npoints,ndim)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def cartesian_to_spherical(xyz,radius):
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    dist = np.sqrt(xy + xyz[:, 2]**2)
    elev = np.arctan2(xyz[:, 2],  np.sqrt(xy)) # for elevation angle defined from XY-plane up
    azim = np.arctan2(xyz[:, 1],  xyz[:, 0])
    elev = [math.degrees(x) for x in elev]
    azim = [math.degrees(x) for x in azim]
    return dist, elev, azim


def main(args):
    mesh_path = args.mesh_path
    mesh_instance = mesh.TriangleMesh(mesh_path=mesh_path)
    mesh_instance.load_pytorch_mesh_from_file()

    light_instance = Lights()
    light_instance.setup_light([args.light_x,  args.light_y,  args.light_z])

    camera_cartesian_points = sample_spherical(args.num_views)
    dist, elev, azim = cartesian_to_spherical(camera_cartesian_points,args.radius)

    dist *= 3
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(args.num_views):
        camera_instance = Camera()
        camera_instance.lookAt(2.7, elev[i], azim[i])

        rasterizer_instance = Rasterizer()
        rasterizer_instance.init_rasterizer(camera_instance.camera)

        shader_instance = Shader()
        shader_instance.setup_shader(camera_instance.camera,  light_instance.light)

        renderer_instance = MeshRenderer(rasterizer=rasterizer_instance.rasterizer,  shader=shader_instance.shader)
        images = renderer_instance(mesh_instance.pytorch_mesh)

        np_image = images[0].cpu().detach().numpy()*255.0
        np_image[:,:,3] = 255.0
        np_image = np_image.astype('uint8')
        pil_image = Image.fromarray(np_image)
        pil_image.save(os.path.join(args.out_dir,str(i)+'.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", help="Path to the mesh file")
    parser.add_argument("--radius", type=float,  help="Sphere Radius")
    parser.add_argument("--light-x", type=float,  help="Light X")
    parser.add_argument("--light-y", type=float,  help="Light Y")
    parser.add_argument("--light-z", type=float,  help="Light Z")
    parser.add_argument("--num-views", type=int, help="Number of views")
    parser.add_argument("--out-dir", help="Path to the output directory")

    args = parser.parse_args()
    main(args)


