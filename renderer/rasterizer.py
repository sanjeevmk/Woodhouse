from pytorch3d.renderer import RasterizationSettings
from pytorch3d.renderer import MeshRasterizer

class Rasterizer:
    def init_rasterizer(self,cameras,image_size=512, blur_radius=0.0, faces_per_pixel=1):
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=blur_radius,
                                                faces_per_pixel=faces_per_pixel)

        self.rasterizer = MeshRasterizer(cameras=cameras,raster_settings=raster_settings)