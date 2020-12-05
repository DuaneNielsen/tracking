from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import gym

from pytorch3d.renderer import \
    (TexturesVertex, MeshRenderer, MeshRasterizer, FoVOrthographicCameras, RasterizationSettings,
     look_at_view_transform, BlendParams)
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.transforms import Translate

from utils import load_blender_stl_mesh
from renderer import CustomFlatShader
from env.runner import EnvRunner, PngCapture
from env.wrappers import ClipState2D, Resize2D
from random import randint
import torch.utils.data
import torchvision.datasets
from torch.autograd import gradcheck
from models.mnn import make_layers
from models.layerbuilder import LayerMetaData
from filter import sample_particles_from_heatmap_2d

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)

""" generate dataset """

pongdir = 'data/Pong_1'


def policy(state):
    """ a random policy to generate actions in Pong"""
    return randint(2, 3)


if not Path(pongdir).exists():
    env = gym.make('Pong-v0')
    env = ClipState2D(env, 0, 24, 210 - 24, 160)
    env = Resize2D(env, h=128, w=128)
    run = EnvRunner(env)
    run.attach_observer('image_cap', PngCapture(pongdir + '/screens'))
    run.episode(policy, render=False)

train_dataset = torchvision.datasets.ImageFolder(
    root=pongdir,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    num_workers=0,
    shuffle=True
)

""" a square mesh """

verts = OrderedDict()
faces = OrderedDict()
colors = OrderedDict()
models = OrderedDict()
particles_per_mesh = OrderedDict()
meshes = OrderedDict()

for i, file in enumerate(Path('./data/meshes').iterdir()):
    name = str(file.stem)
    models[name] = i
    particles_per_mesh[name] = 1
    vert, normal, st, color, face = load_blender_stl_mesh(file)
    V, _ = vert.shape
    verts[name] = vert.to(device)
    faces[name] = face.to(device)
    colors[name] = torch.cat((color.to(device), torch.ones(V, 1, device=device)), dim=1)


def create_scene(hws, alphas, N):
    batch = []
    for i in range(N):
        scene = []
        for mesh_name in models:
            hw = hws[mesh_name]
            alpha = alphas[mesh_name]
            N, K, _ = hw.shape
            for k in range(K):
                c = colors[mesh_name].clone()
                c[..., 3] = alpha[i, k]
                textures = TexturesVertex(verts_features=[c])
                m = Meshes(
                    verts=[verts[mesh_name].clone()],
                    faces=[faces[mesh_name].clone()],
                    textures=textures
                )
                # m = meshes[mesh_name].clone().detach().to(device)
                t = Translate(y=hw[i, k, 0], x=hw[i, k, 1], z=torch.zeros(1, device=device), device=str(device))
                m = m.update_padded(t.transform_points(m.verts_padded()))
                scene += [m]
        batch += [join_meshes_as_scene(scene)]
    batch = join_meshes_as_batch(batch)
    return batch


class Tracker(nn.Module):
    def __init__(self, device):
        super().__init__()

        distance, elevation, azimuth = 30, 0.0, 0
        self.R, self.T = look_at_view_transform(distance, elevation, azimuth, device=device)
        meta = LayerMetaData(input_shape=(3, 128, 128))
        self.tracker, _ = make_layers(['C:3', 16, 16, 16, 8],
                                         type='resnet', meta=meta)

        self.tracker = self.tracker.to(device)
        cameras = FoVOrthographicCameras(device=device,
                                         max_x=64.0, max_y=64.0,
                                         min_x=-64.0, min_y=-64.0,
                                         scale_xyz=((1, 1, 1),))

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=RasterizationSettings(
                    image_size=128,
                    blur_radius=1e-9,
                    faces_per_pixel=10,
                )
            ),
            shader=CustomFlatShader(
                device=str(device),
                cameras=cameras
            )
        )

        self.particle_config = OrderedDict({
            'background': 2,
            'green_paddle': 2,
            'top_wall': 2,
            'bottom_wall': 2,
            'puck': 2,
            'red_paddle': 2
        })

    def forward(self, image):

        heatmap = self.tracker(image)
        hws, alphas = sample_particles_from_heatmap_2d(heatmap[:, 0:6], k=self.particle_config, device=device,
                                         h_min=-64.0, h_max=64.0,
                                         w_min=-64, w_max=64.0)

        # dummy function here N, K, D
        #hws = OrderedDict()
        #alphas = OrderedDict()

        N = 8

        # # N, K co-ords and alphas
        for name in models:
            if name == 'background':
                hws[name] = torch.zeros(N, 2, 2, device=device)
        #     else:
        #         hws[name] = torch.zeros(N, 2, 2, device=device) * 10.0
        #     alphas[name] = torch.ones(N, 2, device=device, requires_grad=True)

        scene = create_scene(hws, alphas, N)
        return self.renderer(meshes_world=scene, R=self.R, T=self.T), heatmap


tracker = Tracker(device=device)


class Plot:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 12))
        self.axe = [self.fig.add_subplot(4, 4, i + 1) for i in range(16)]

    def update_heatmap(self, heatmap):
        for i, image in enumerate(heatmap):
            self.axe[i].clear()
            self.axe[i].imshow(image.detach().cpu(), cmap='hot', interpolation='nearest')

    def update_render(self, render):
        for i, image in enumerate(render):
            self.axe[i+8].clear()
            self.axe[i+8].imshow(image.detach().cpu(), interpolation='nearest')

    def show(self, pause):
        plt.pause(pause)


heatmap_plot = Plot()


for batch, _ in train_loader:
    render, heatmap = tracker(batch.to(device))
    heatmap_plot.update_heatmap(heatmap[0])
    heatmap_plot.update_render(render)
    heatmap_plot.show(2)