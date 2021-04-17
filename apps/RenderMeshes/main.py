import argparse
from ...geometry import mesh


def main(args):
    mesh_path = args.mesh_path
    mesh_instance = mesh.TriangleMesh(mesh_path=mesh_path)
    mesh_instance.load_trimesh_from_file()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_path",help="Path to the mesh file")

    args = parser.parse_args()
    main(args)


