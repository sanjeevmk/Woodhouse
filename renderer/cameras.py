from pytorch3d.renderer import look_at_view_transform,FoVPerspectiveCameras

class Camera:
    def __init__(self,camera_type='fov',device='cuda'):
        self.camera_type = camera_type
        self.device = device

    def lookAt(self,dist=0.0,elev=0.0,azim=0.0):
        R,T = look_at_view_transform(dist,elev,azim)

        if self.camera_type == 'fov':
            self.camera = FoVPerspectiveCameras(device=self.device,R=R,T=T)

    def getLocation(self):
        location = self.camera.get_camera_center()

        return location

