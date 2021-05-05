from pytorch3d.renderer import PointLights

class Lights:
    def __init__(self,device='cuda',light_type='point'):
        self.device = device
        self.light_type = light_type

    def setup_light(self,position):
        if self.light_type == 'point':
            self.light = PointLights(device=self.device,location=[position])