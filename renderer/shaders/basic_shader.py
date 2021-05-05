from pytorch3d.renderer import SoftPhongShader

class Shader:
    def __init__(self,device='cuda',shader_type='soft_phong'):
        self.device = device
        self.shader_type = shader_type

    def setup_shader(self,cameras,lights):
        if self.shader_type == 'soft_phong':
            self.shader = SoftPhongShader(device=self.device,cameras=cameras,lights=lights)