import torch
from visdom import Visdom
from typing import List

def visualize_image_outputs(
        validation_images : List, viz: Visdom, visdom_env: str
):
    ims = torch.cat(validation_images,2)
    viz.image(
        ims,
        env=visdom_env,
        win="images",
        opts={"title": "validation_images"},
    )

def visualize_image_list_vertically(
        images: List, viz: Visdom, visdom_env: str, n_rows: int = 1
):
    images = torch.cat(images,1)
    viz.image(
        images,
        env=visdom_env,
        win="images",
        opts={"title": "validation_images"},
    )
