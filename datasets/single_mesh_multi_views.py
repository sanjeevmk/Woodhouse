from torch.utils.data import Dataset
import trimesh
import json
from PIL import Image
import os
import numpy as np
import math
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from typing import List


class CowMultiViews(Dataset):
    EXT = '.png'
    def __init__(self,mesh_path,views_folder,texture_path,params_file="",views:np.ndarray=[],param_vectors:np.ndarray=[],image_size:List=[512,512]):
        self.mesh_path = mesh_path
        self.texture_path = texture_path
        self.views_folder = views_folder
        self.trimesh_object = self.get_trimesh()
        self.pytorch_mesh = self.get_pytorch_mesh()
        self.image_height = image_size[0]
        self.image_width = image_size[1]
        self.params_file = params_file

        if len(views) == 0:
            assert params_file,"CowMultiViews: If views not given, provide params file"
            self.params = json.loads(open(params_file,'r').read())
            self.views,self.param_vectors = self.load_views_and_params()
            assert (len(self.views) == len(self.param_vectors)),"CowMultiViews: Number of views should match number " \
                                                                "of parameters given"
        else:
            self.views = views
            assert (len(param_vectors)>0),"CowMultiViews: Views given, but param vectors not given"
            self.param_vectors = param_vectors

    def get_trimesh(self) -> trimesh.Trimesh :
        return trimesh.load(self.mesh_path,process=False)

    def get_pytorch_mesh(self) -> Meshes:
        return load_objs_as_meshes([self.mesh_path])

    def get_verts(self):
        return np.array(self.trimesh_object.vertices)

    def get_edges(self):
        return np.array(self.trimesh_object.edges)

    def get_vert_normals(self):
        return np.array(self.trimesh_object.vertex_normals)

    def get_texture(self):
        np_texture = np.asarray(Image.open(self.texture_path))
        return np_texture

    def load_views_and_params(self):
        param_vectors = []
        views = []
        for image_index in self.params:
            image_path = os.path.join(self.views_folder,image_index+CowMultiViews.EXT)
            pil_view = Image.open(image_path)
            np_view = np.asarray(pil_view)[:,:,:3]  #discarding opacity

            view_param_vector = []
            dist = self.params[image_index]['camera']['dist']
            elev = self.params[image_index]['camera']['elev']/(2*math.pi)
            azim = self.params[image_index]['camera']['azim']/(2*math.pi)

            view_param_vector.append(dist)
            view_param_vector.append(elev)
            view_param_vector.append(azim)

            light_ambient = self.params[image_index]['light']['ambient']
            light_diffuse = self.params[image_index]['light']['diffuse']
            light_specular = self.params[image_index]['light']['specular']
            light_direction = self.params[image_index]['light']['direction']

            view_param_vector.extend(light_ambient+ light_diffuse + light_specular + light_direction)

            param_vectors.append(view_param_vector)
            views.append(np_view)
        return np.array(views),np.array(param_vectors)

    @staticmethod
    def random_split_dataset(dataset_instance,train_fraction=0.7,validation_fraction=0.2):
        indices = list(range(len(dataset_instance.views)))
        import random
        random.seed(0)
        random.shuffle(indices)
        train_split_point = int(train_fraction*len(indices))
        train_indices = indices[:train_split_point]

        validation_split_point = train_split_point + int(validation_fraction*len(indices))
        validation_indices = indices[train_split_point:validation_split_point]

        test_indices = indices[validation_split_point:]

        train_dataset = CowMultiViews(dataset_instance.mesh_path,dataset_instance.views_folder,
                                      dataset_instance.texture_path,dataset_instance.params_file,
                                      views=dataset_instance.views[train_indices,:,:,:],
                                      param_vectors=dataset_instance.param_vectors[train_indices,:])

        validation_dataset = CowMultiViews(dataset_instance.mesh_path, dataset_instance.views_folder,
                                           dataset_instance.texture_path,dataset_instance.params_file,
                                           views=dataset_instance.views[validation_indices, :, :, :],
                                           param_vectors=dataset_instance.param_vectors[validation_indices, :])

        test_dataset = CowMultiViews(dataset_instance.mesh_path, dataset_instance.views_folder,
                                     dataset_instance.texture_path,dataset_instance.params_file,
                                     views=dataset_instance.views[test_indices, :, :, :],
                                     param_vectors=dataset_instance.param_vectors[test_indices, :])

        return train_dataset,validation_dataset,test_dataset

    def __len__(self):
        return len(self.views)

    def __getitem__(self,idx):
        view = self.views[idx,:,:]
        view = view.astype(np.float64)
        view /= 255.0

        param_vectors = self.param_vectors[idx,:]

        return view, param_vectors
