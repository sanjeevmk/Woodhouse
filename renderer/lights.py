from typing import Tuple
from pytorch3d.renderer import PointLights,DirectionalLights

class Lights:
    def __init__(self,light_type:str , ambient_color:Tuple, diffuse_color:Tuple, specular_color:Tuple):
        self.light_type = light_type
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.light = None

    def setup_light(self,**kwargs):
        if self.light_type == 'point':
            if 'position' in kwargs:
                position = kwargs['position']
                self.light = PointLights(ambient_color=self.ambient_color, diffuse_color=self.diffuse_color,
                                         specular_color=self.specular_color, location=[position],device='cuda')

        if self.light_type == 'directional':
            if 'direction' in kwargs:
                direction = kwargs['direction']
                self.light = DirectionalLights(ambient_color=self.ambient_color,diffuse_color=self.diffuse_color,
                                               specular_color=self.specular_color, direction=direction,device='cuda')