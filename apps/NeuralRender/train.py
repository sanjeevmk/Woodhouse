import torch
import numpy as np
import hydra
from omegaconf import DictConfig
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd+"/../woodhouse_lib/")
from datasets.single_mesh_multi_views import CowMultiViews
from torch.utils.data import DataLoader
from visdom import Visdom
from utils.stats import Stats
import collections
import pickle
from utils.visualize_outputs import visualize_image_outputs
from util_networks.image_translator import ImageTranslator
from renderer.cameras import Camera
from renderer.rasterizer import Rasterizer
from pytorch3d.ops import interpolate_face_attributes
import math

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

    mesh_verts = train_dataset.get_verts()
    mesh_edges = train_dataset.get_edges()
    mesh_vert_normals = train_dataset.get_vert_normals()
    mesh_texture = train_dataset.get_texture()
    pytorch_mesh = train_dataset.pytorch_mesh.cuda()
    face_attrs = train_dataset.get_faces_as_vertex_matrices(features_list=['random'],num_random_dims=cfg.training.feature_dim)

    torch_verts = torch.from_numpy(np.array(mesh_verts)).float().cuda()
    torch_edges = torch.from_numpy(np.array(mesh_edges)).long().cuda()
    torch_normals = torch.from_numpy(np.array(mesh_vert_normals)).float().cuda()
    torch_texture = torch.from_numpy(np.array(mesh_texture)).float().cuda()
    torch_texture = torch.unsqueeze(torch_texture,0)
    torch_face_attrs = torch.tensor(np.array(face_attrs),requires_grad=True).float().cuda()
    torch_face_attrs = torch.nn.Parameter(torch_face_attrs)

    train_dataloader = DataLoader(train_dataset,batch_size=cfg.training.batch_size,shuffle=True,num_workers=4)
    validation_dataloader = DataLoader(validation_dataset,batch_size=cfg.training.batch_size,shuffle=True,num_workers=4)

    image_translator = ImageTranslator(input_dim=cfg.training.feature_dim,output_dim=3,
                                   image_size=tuple(cfg.data.image_size)).cuda()

    mse_loss = torch.nn.MSELoss()

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        list(image_translator.parameters())+[torch_face_attrs],
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

    for epoch in range(cfg.optimizer.max_epochs):
        image_translator.train()
        stats.new_epoch()
        for iteration,data in enumerate(train_dataloader):
            optimizer.zero_grad()

            views,param_vectors = data
            views = views.float().cuda()
            param_vectors = param_vectors.float().cuda()
            camera_instance = Camera()
            camera_instance.lookAt(param_vectors[0][0],math.degrees(param_vectors[0][1]),math.degrees(param_vectors[0][2]))

            rasterizer_instance = Rasterizer()
            rasterizer_instance.init_rasterizer(camera_instance.camera)
            fragments = rasterizer_instance.rasterizer(pytorch_mesh)
            pix_to_face = fragments.pix_to_face
            bary_coords = fragments.bary_coords

            pix_features = torch.squeeze(interpolate_face_attributes(pix_to_face,bary_coords,torch_face_attrs),3)
            predicted_render = image_translator(pix_features,torch_texture)

            loss = mse_loss(predicted_render,views)
            loss.backward()
            optimizer.step()

            # Update stats with the current metrics.
            stats.update(
                {"mse_loss": float(loss)},
                stat_set="train",
            )

            if iteration % cfg.stats_print_interval == 0:
                stats.print(stat_set="train")

        # Adjust the learning rate.
        #lr_scheduler.step()

        # Validation
        if epoch % cfg.validation_epoch_interval == 0: # and epoch > 0:

            # Sample a validation camera/image.
            val_batch = next(validation_dataloader.__iter__())
            views, param_vectors= val_batch
            views = views.float().cuda()
            param_vectors = param_vectors.float().cuda()

            # Activate eval mode of the model (allows to do a full rendering pass).
            image_translator.eval()
            with torch.no_grad():
                camera_instance = Camera()
                camera_instance.lookAt(param_vectors[0][0], math.degrees(param_vectors[0][1]), math.degrees(param_vectors[0][2]))

                rasterizer_instance = Rasterizer()
                rasterizer_instance.init_rasterizer(camera_instance.camera)
                fragments = rasterizer_instance.rasterizer(pytorch_mesh)
                pix_to_face = fragments.pix_to_face
                bary_coords = fragments.bary_coords

                pix_features = torch.squeeze(interpolate_face_attributes(pix_to_face, bary_coords, torch_face_attrs), 3)
                #pix_features = pix_features.permute(0, 3, 1, 2)
                predicted_render = image_translator(pix_features,torch_texture)
                loss = mse_loss(predicted_render,views)


            # Update stats with the validation metrics.
            stats.update({"mse_loss":loss}, stat_set="val")
            stats.print(stat_set="val")

            if viz is not None:
                # Plot that loss curves into visdom.
                stats.plot_stats(
                    viz=viz,
                    visdom_env=cfg.visualization.visdom_env,
                    plot_file=None,
                )
                # Visualize the intermediate results.
                render_max = torch.max(predicted_render)
                visualize_image_outputs(
                    validation_images = [views[0].permute(2,0,1),predicted_render[0].permute(2,0,1)],viz=viz,visdom_env=cfg.visualization.visdom_env
                )

            # Set the model back to train mode.
            image_translator.train()

        # Checkpoint.
        if (
                epoch % cfg.checkpoint_epoch_interval == 0
                and len(cfg.checkpoint_path) > 0
                and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")
            data_to_store = {
                "model": image_translator.state_dict(),
                "features" : torch_face_attrs,
                "optimizer": optimizer.state_dict(),
                "stats": pickle.dumps(stats),
            }
            torch.save(data_to_store, checkpoint_path)

if __name__ == "__main__":
    main()
