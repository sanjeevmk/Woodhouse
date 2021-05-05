import argparse
from input_representation import mesh
from renderer.cameras import Camera
from renderer.rasterizer import Rasterizer
from renderer.lights import Lights
from renderer.shaders.basic_shader import Shader
from pytorch3d.renderer import MeshRenderer
from PIL import Image


def main(args):
    mesh_path = args.mesh_path
    mesh_instance = mesh.TriangleMesh(mesh_path=mesh_path)
    mesh_instance.load_pytorch_mesh_from_file()

    camera_instance = Camera()
    camera_instance.lookAt(args.dist,  args.elev,  args.azim)

    light_instance = Lights()
    light_instance.setup_light([args.light_x,  args.light_y,  args.light_z])

    rasterizer_instance = Rasterizer()
    rasterizer_instance.init_rasterizer(camera_instance.camera)

    shader_instance = Shader()
    shader_instance.setup_shader(camera_instance.camera,  light_instance.light)

    renderer_instance = MeshRenderer(rasterizer=rasterizer_instance.rasterizer,  shader=shader_instance.shader)
    images = renderer_instance(mesh_instance.pytorch_mesh)

    np_image = images[0].cpu().detach().numpy()*255.0
    np_image = np_image.astype('uint8')
    pil_image = Image.fromarray(np_image)
    pil_image.save(args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-path", help="Path to the mesh file")
    parser.add_argument("--dist", type=float, help="Camera Distance")
    parser.add_argument("--elev", type=float, help="Camera Elevation")
    parser.add_argument("--azim", type=float, help="Camera Azimuth")
    parser.add_argument("--light-x", type=float,  help="Light X")
    parser.add_argument("--light-y", type=float,  help="Light Y")
    parser.add_argument("--light-z", type=float,  help="Light Z")
    parser.add_argument("--out-path", help="Path to the output image")

    args = parser.parse_args()
    main(args)


