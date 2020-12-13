from collections import OrderedDict, deque
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import matplotlib.patches as patches
import gym

from pytorch3d.renderer import \
    (TexturesVertex, MeshRenderer, MeshRasterizer, FoVOrthographicCameras, RasterizationSettings,
     look_at_view_transform, BlendParams, SoftSilhouetteShader, HardFlatShader)
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.transforms import Translate

from utils import load_blender_ply_mesh
from renderer import CustomFlatShader
from env.runner import EnvRunner, PngCapture
from env.wrappers import ClipState2D, Resize2D, ApplyFunc
from random import randint
import torch.utils.data
import torchvision.datasets
import torchvision.utils
from torch.autograd import gradcheck
from models.mnn import make_layers
from models.layerbuilder import LayerMetaData
from filter import sample_particles_from_heatmap_2d
import matplotlib.gridspec as gridspec
from utils import BoundingBoxes
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
torch.autograd.set_detect_anomaly(True)
import numpy as np
import cv2
import engine


class MinimalFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(
            self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        return texels[..., 0, :]


if __name__ == '__main__':

    world = engine.World()

    objects = []
    for i, file in enumerate(Path('./data/meshes').iterdir()):
        name = str(file.stem)
        vert, normal, st, color, face = load_blender_ply_mesh(file)
        world.add_mesh(name, vert, face, color)
        objects += [name]

    hws, alphas = {}, {}

    N, K, = 2, 1

    scenes_spec = []
    for _ in range(N):
        scene_obs = {}
        for obj in objects:
            scene_obs[obj] = obj
        scenes_spec += [scene_obs]

    world.create_scenes(scenes_spec)

    def gridnum(i, ncols):
        return i // ncols, i % ncols

    plt.ion()
    fig = plt.figure(figsize=(10, 5))
    spec = plt.GridSpec(ncols=N // 2, nrows=2, figure=fig)
    ax = [fig.add_subplot(spec[gridnum(i, ncols=N // 2)]) for i in range(N)]
    plt.tight_layout()

    for b in range(50):

        transforms = []
        for _ in range(N):
            model_matrices = []
            for _ in objects:
                xy = torch.rand(2) * 128 - 64
                model_matrices += [Translate(x=xy[0], y=xy[1], z=0)]
            transforms += [model_matrices]

        distance = 30
        elevation = 0.0
        azimuth = 0
        R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

        cameras = FoVOrthographicCameras(device=device,
                                         max_x=64.0, max_y=64.0,
                                         min_x=-64.0, min_y=-64.0,
                                         scale_xyz=((1, 1, 1),), R=R, T=T)

        background_color = torch.tensor([144 / 255, 71 / 255, 16 / 255], device=device)
        blend_params = BlendParams(background_color=background_color)

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=128,
                    blur_radius=1e-9,
                    faces_per_pixel=12,
                )
            ),
            shader=MinimalFlatShader(
                device=str(device),
                cameras=cameras,
                blend_params=blend_params
            )
        )

        world.update_scenes(transforms)
        batch, labels, bbox = world.batch(), world.labels(), world.bounding_boxes(cameras, screen_size=(128, 128))
        image = renderer(batch)
        image = torch.round(image * 255)

        for i in range(N):
            ax[i].clear()
            ax[i].imshow(image[i].numpy().astype(np.uint8))
            for bb, label in zip(bbox[i], labels[i]):
                ax[i].add_patch(bb.get_patch(margin=(1, 1)))
                ax[i].text(bb.top_right.x, bb.top_right.y, world.object_index[label].name, color='m')
        plt.pause(10.0)
