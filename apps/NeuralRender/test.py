import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import os
from datasets.single_mesh_multi_views import CowMultiViews
from torch.utils.data import DataLoader
from renderer.shaders.graph_conv_shaders import MeshRenderer as GraphRenderer
from pytorch3d.structures import Meshes
from visdom import Visdom
from utils.stats import Stats
import collections
import pickle
from utils.visualize_outputs import visualize_image_outputs
from util_networks.image_encoder import ImageEncoder

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

    mesh_verts = train_dataset.get_verts()
    mesh_edges = train_dataset.get_edges()
    mesh_vert_normals = train_dataset.get_vert_normals()
    mesh_texture = train_dataset.get_texture()

    feature_size = train_dataset.param_vectors.shape[1]

    torch_verts = torch.from_numpy(np.array(mesh_verts)).float().cuda()
    torch_edges = torch.from_numpy(np.array(mesh_edges)).long().cuda()
    torch_normals = torch.from_numpy(np.array(mesh_vert_normals)).float().cuda()
    torch_texture = torch.from_numpy(np.array(mesh_texture)).float().cuda()
    torch_texture = torch.unsqueeze(torch_texture.permute(2,0,1),0)

    train_dataloader = DataLoader(train_dataset,batch_size=cfg.training.batch_size,shuffle=True,num_workers=4)
    validation_dataloader = DataLoader(validation_dataset,batch_size=cfg.training.batch_size,shuffle=True,num_workers=4)

    graph_renderer = GraphRenderer(input_dim=6+train_dataset.param_vectors.shape[1]+cfg.training.texture_feature_dim,
                                   image_size=tuple(cfg.data.image_size)).cuda()
    texture_encoder = ImageEncoder(output_dim=cfg.training.texture_feature_dim).cuda()

    mse_loss = torch.nn.MSELoss()

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        list(graph_renderer.parameters()) + list(texture_encoder.parameters()),
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

    graph_renderer.eval()
    texture_encoder.eval()
    stats.new_epoch()


    image_list = []
    for iteration,data in enumerate(train_dataloader):
        if iteration==10:
            break
        optimizer.zero_grad()
        texture_feature = texture_encoder(torch_texture)

        views,param_vectors = data
        views = views.float().cuda()
        param_vectors = param_vectors.float().cuda()
        param_vectors = torch.repeat_interleave(param_vectors,torch_verts.size()[0],0)
        texture_feature_repeat = torch.repeat_interleave(texture_feature,torch_verts.size()[0],0)
        torch_featured_verts = torch.cat([torch_verts,torch_normals,texture_feature_repeat,param_vectors],1)
        predicted_render = graph_renderer(torch_featured_verts,torch_edges)

        viz_display_image = torch.cat([predicted_render,views.permute(0,3,1,2)],0)
        image_list.append(viz_display_image)

    if viz is not None:
        visualize_image_outputs(
            validation_images = image_list,viz=viz,visdom_env=cfg.visualization.visdom_env
        )