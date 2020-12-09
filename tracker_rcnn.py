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
from torch.autograd import gradcheck
from models.mnn import make_layers
from models.layerbuilder import LayerMetaData
from filter import sample_particles_from_heatmap_2d
import matplotlib.gridspec as gridspec
from utils import BoundingBoxes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
torch.autograd.set_detect_anomaly(True)
import numpy as np
import cv2


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


class World:
    def __init__(self, device='cpu', add_alpha=False):
        self.device = device
        self.verts = OrderedDict()
        self.faces = OrderedDict()
        self.colors = OrderedDict()
        self.models = OrderedDict()
        self.meshes = OrderedDict()
        self.add_alpha = add_alpha

    def add_mesh(self, name, vert, face, color):
        """

        :param name: unique label for the mesh
        :param vert: V, 3 vert list
        :param face: F, 3 LongTensor
        :param color: V, 3 vertex colors
        """
        V, _ = vert.shape
        self.models[name] = 1
        self.verts[name] = vert.to(self.device)
        self.faces[name] = face.to(self.device)
        if self.add_alpha:
            self.colors[name] = torch.cat((color.to(self.device), torch.ones(V, 1, device=self.device)), dim=1)
        else:
            self.colors[name] = color.to(self.device)

    def create_scene(self, hws, alphas, N):
        batch = []
        bounding_boxes = []
        bb_labels = []
        for i in range(N):
            scene = []
            bounding_box = []
            bounding_box_label = []
            for mesh_name in self.models:
                hw = hws[mesh_name]
                alpha = alphas[mesh_name]
                N, K, _ = hw.shape
                for k in range(K):
                    c = self.colors[mesh_name].clone()
                    if self.add_alpha:
                        c[..., 3] = alpha[i, k]

                    textures = TexturesVertex(verts_features=[c])
                    m = Meshes(
                        verts=[self.verts[mesh_name].clone()],
                        faces=[self.faces[mesh_name].clone()],
                        textures=textures
                    )

                    t = Translate(y=hw[i, k, 0], x=hw[i, k, 1], z=torch.zeros(1, device=self.device),
                                  device=str(self.device))
                    v = t.transform_points(m.verts_padded())
                    m = m.update_padded(v)
                    scene += [m]
                    bounding_box += [m.get_bounding_boxes()]
                    bounding_box_label += [mesh_name]

            batch += [join_meshes_as_scene(scene)]
            bounding_boxes += [bounding_box]
            bb_labels += [bounding_box_label]
        batch = join_meshes_as_batch(batch)
        return batch, bounding_boxes, bb_labels


if __name__ == '__main__':

    world = World(add_alpha=False)

    for i, file in enumerate(Path('./data/meshes').iterdir()):
        name = str(file.stem)
        vert, normal, st, color, face = load_blender_ply_mesh(file)
        world.add_mesh(name, vert, face, color)

    hws, alphas = {}, {}

    N, K, = 10, 1


    def gridnum(i, ncols):
        return i // ncols, i % ncols


    plt.ion()
    fig = plt.figure(figsize=(10, 10), constrained_layout=False)
    spec = plt.GridSpec(ncols=N // 2, nrows=2, figure=fig)
    ax = [fig.add_subplot(spec[gridnum(i, ncols=N // 2)]) for i in range(N)]

    for _ in range(50):
        for name in world.models:
            hws[name] = torch.rand(N, K, 2) * 128 - 64
            alphas[name] = torch.ones(N, K)

        scene, bounding_boxes, labels = world.create_scene(hws, alphas, N)

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

        boxes_screen_co_ords = []

        box_batch = []
        for n in range(N):
            box_image = []
            for b in range(len(bounding_boxes[n])):
                label, box = labels[n][b], bounding_boxes[n][b]
                box_screen = cameras.transform_points_screen(box[:, :, :].permute(0, 2, 1), torch.tensor([[128, 128]]))
                box_screen = box_screen[:, :, 0:2]
                box_image += [box_screen]
            box_batch += [box_image]

        image = renderer(scene)

        for i in range(N):
            ax[i].clear()
            ax[i].imshow(image[i].clamp(0, 1))
            for box in box_batch[i]:
                bb = BoundingBoxes(box)
                #min_x, min_y = box[0, 0, 0], box[0, 0, 1]
                boxes_rect = patches.Rectangle(bb.mpl_anchor(0), height=bb.height(0), width=bb.width(0), linewidth=1, edgecolor='r', facecolor='none')
                ax[i].add_patch(boxes_rect)
        plt.pause(0.2)
