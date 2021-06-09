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
from input_representation.mesh import TriangleMesh

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

    def get_faces(self):
        return np.array(self.trimesh_object.faces)

    def unit_normalize(self):
        triangle_mesh = TriangleMesh(self.mesh_path)
        triangle_mesh.load_trimesh_from_file()
        triangle_mesh.load_pytorch_mesh_from_file()
        triangle_mesh = triangle_mesh.unit_normalize()
        self.trimesh_object = triangle_mesh.mesh
        self.pytorch_mesh = triangle_mesh.pytorch_mesh

    def get_faces_as_vertex_matrices(self,features_list=[],num_random_dims=-1):
        faces = self.get_faces()
        verts = self.get_verts()
        normals = self.get_vert_normals()

        faces_attr = []
        for i in range(faces.shape[0]):
            face_attr = []
            for j in range(3):
                vert_index = faces[i][j]
                coord_feature = []
                normal_feature = []
                random_feature = []
                if 'coord' in features_list:
                    coord_feature = [verts[vert_index][c] for c in range(3)]
                if 'normal' in features_list:
                    normal_feature = [normals[vert_index][c] for c in range(3)]
                if 'random' in features_list:
                    random_feature = np.random.normal(size=num_random_dims).tolist()
                vertex_feature = coord_feature + normal_feature + random_feature
                face_attr.append(vertex_feature)
            faces_attr.append(face_attr)

        return np.array(faces_attr)

    def get_vert_normals(self):
        return np.array(self.trimesh_object.vertex_normals)

    def get_texture(self,image_size=[512,512]):
        pil_texture = Image.open(self.texture_path)
        pil_texture = pil_texture.resize((image_size[0],image_size[1]))
        np_texture = np.asarray(pil_texture)
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
            elev = self.params[image_index]['camera']['elev']
            azim = self.params[image_index]['camera']['azim']
            camera_location = self.params[image_index]['camera']['position']

            view_param_vector.append(dist)
            view_param_vector.append(elev)
            view_param_vector.append(azim)
            view_param_vector.extend(camera_location)

            light_ambient = self.params[image_index]['light']['ambient']
            light_diffuse = self.params[image_index]['light']['diffuse']
            light_specular = self.params[image_index]['light']['specular']
            light_location = self.params[image_index]['light']['position']

            #view_param_vector.extend(light_ambient+ light_diffuse + light_specular + light_direction)
            view_param_vector.extend(light_location)

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
