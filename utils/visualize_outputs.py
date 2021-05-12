import torch
from visdom import Visdom
from typing import List

def visualize_image_outputs(
        validation_images : List, viz: Visdom, visdom_env: str
):
    ims = torch.cat(validation_images,1)
    viz.image(
        ims,
        env=visdom_env,
        win="images",
        opts={"title": "validation_images"},
    )