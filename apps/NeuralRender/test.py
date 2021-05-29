import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import os
from datasets.single_mesh_multi_views import CowMultiViews
from torch.utils.data import DataLoader,Subset
from renderer.shaders.graph_conv_shaders import MeshRenderer as GraphRenderer
from pytorch3d.structures import Meshes
from visdom import Visdom
from utils.stats import Stats
import collections
import pickle
from utils.visualize_outputs import visualize_image_outputs,visualize_image_list_vertically
from util_networks.image_translator import ImageTranslator
from renderer.cameras import Camera
from renderer.rasterizer import Rasterizer
from pytorch3d.ops import interpolate_face_attributes
import math
import random


CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="cow")
def main(cfg: DictConfig):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    obj_path = cfg.data.obj_path
    texture_path = cfg.data.texture_path
    views_folder = cfg.data.views_folder
    params_file = os.path.join(views_folder,"params.json")
    dataset = CowMultiViews(obj_path,views_folder,texture_path,params_file=params_file)

    train_dataset, validation_dataset, test_dataset = CowMultiViews.random_split_dataset(dataset,
                                                                                         train_fraction=0.7,
                                                                                         validation_fraction=0.2)

    del dataset
    train_dataset.unit_normalize()
    validation_dataset.unit_normalize()
    test_dataset.unit_normalize()


    mesh_verts = test_dataset.get_verts()
    mesh_edges = test_dataset.get_edges()
    mesh_vert_normals = test_dataset.get_vert_normals()
    mesh_texture = test_dataset.get_texture()
    pytorch_mesh = test_dataset.pytorch_mesh.cuda()
    face_attrs = test_dataset.get_faces_as_vertex_matrices()

    feature_size = test_dataset.param_vectors.shape[1]

    torch_verts = torch.from_numpy(np.array(mesh_verts)).float().cuda()
    torch_edges = torch.from_numpy(np.array(mesh_edges)).long().cuda()
    torch_normals = torch.from_numpy(np.array(mesh_vert_normals)).float().cuda()
    torch_texture = torch.from_numpy(np.array(mesh_texture)).float().cuda()
    torch_texture = torch.unsqueeze(torch_texture.permute(2,0,1),0)
    torch_face_attrs = torch.from_numpy(np.array(face_attrs)).float().cuda()

    subset_indices = [82] #random.sample(list(range(len(test_dataset))),1)
    test_dataloader = Subset(test_dataset,subset_indices)
    print(subset_indices,len(test_dataloader))

    image_translator = ImageTranslator(input_dim=6,output_dim=3,
                                       image_size=tuple(cfg.data.image_size)).cuda()


    mse_loss = torch.nn.MSELoss()

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        image_translator.parameters(),
        lr=cfg.optimizer.lr,
    )

    stats = None
    start_epoch = 0
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), cfg.checkpoint_path)

    # Init the stats object.
    if stats is None:
        stats = Stats(
            ["mse_loss", "sec/it"],
        )

    # Learning rate scheduler setup.

    # Following the original code, we use exponential decay of the
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda(epoch):
        return cfg.optimizer.lr_scheduler_gamma ** (
                epoch / cfg.optimizer.lr_scheduler_step_size
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    # Initialize the cache for storing variables needed for visulization.
    visuals_cache = collections.deque(maxlen=cfg.visualization.history_size)

    # Init the visualization visdom env.
    if cfg.visualization.visdom:
        viz = Visdom(
            server=cfg.visualization.visdom_server,
            port=cfg.visualization.visdom_port,
            use_incoming_socket=False,
        )
    else:
        viz = None

    loaded_data = torch.load(checkpoint_path)

    image_translator.load_state_dict(loaded_data["model"], strict=False)
    image_translator.eval()
    stats.new_epoch()

    image_list = []
    for iteration,data in enumerate(test_dataloader):
        print(iteration)
        optimizer.zero_grad()

        views,param_vectors = data
        views = torch.unsqueeze(torch.from_numpy(views),0)
        param_vectors = torch.unsqueeze(torch.from_numpy(param_vectors),0)
        views = views.float().cuda()
        param_vectors = param_vectors.float().cuda()
        camera_instance = Camera()
        camera_instance.lookAt(param_vectors[0][0], math.degrees(param_vectors[0][1]),
                               math.degrees(param_vectors[0][2]))

        rasterizer_instance = Rasterizer()
        rasterizer_instance.init_rasterizer(camera_instance.camera)
        fragments = rasterizer_instance.rasterizer(pytorch_mesh)
        pix_to_face = fragments.pix_to_face
        bary_coords = fragments.bary_coords

        pix_features = torch.squeeze(interpolate_face_attributes(pix_to_face, bary_coords, torch_face_attrs), 3)
        param_matrix = torch.zeros(pix_features.size()[0], pix_features.size()[1], pix_features.size()[2],
                                   param_vectors.size()[1]).float().cuda()
        param_matrix[:, :, :, :] = param_vectors
        image_features = pix_features  # torch.cat([pix_features,param_matrix],3)
        predicted_render = image_translator(image_features, torch_texture)

        image_list = [views[0].permute(2,0,1),predicted_render[0].permute(2,0,1)]

    if viz is not None:
        visualize_image_outputs(
            validation_images = image_list,viz=viz,visdom_env=cfg.visualization.visdom_env
        )

if __name__ == "__main__":
    main()
