import trimesh
from ..utils import checks

class TriangleMesh:
    """
    Triangle mesh representation incorporating both Pytorch and Trimesh mesh representations.

    A mesh can have a neural context or a regular non-neural context. The neural context requires use of Pytorch mesh
    and its corresponding utilities, while the non-neural context requires a pythonic representation of the mesh. We use
    trimesh for the latter. This class builds on top of both these mesh contexts by providing a wrapper to
    interact with both.
    """

    def __init__(self, mesh_path="",
                 mesh: trimesh.Trimesh = None):
        if mesh_path:
            checks.check_file_exists(mesh_path)
            self.mesh_path = mesh_path
        if mesh is not None:
            self.mesh = mesh

    @staticmethod
    def load(mesh_path):
        """
        Loads mesh from path as trimesh.Trimesh into self.mesh

        :param mesh_path: Path to mesh file
        :return: TriangleMesh
        """

        checks.check_file_exists(mesh_path)
        mesh = trimesh.load(mesh_path, process=False)
        return TriangleMesh(mesh_path=mesh_path, mesh=mesh)

    def load_trimesh_from_file(self):
        """
        Loads from trimesh object from path attribute
        """
        self.mesh = trimesh.load(self.mesh_path,process=False)