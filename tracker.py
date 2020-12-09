from collections import OrderedDict, deque
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import gym

from pytorch3d.renderer import \
    (TexturesVertex, MeshRenderer, MeshRasterizer, FoVOrthographicCameras, RasterizationSettings,
     look_at_view_transform, BlendParams, SoftSilhouetteShader)
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.autograd.set_detect_anomaly(True)
import numpy as np
import cv2


def remove_background(state):
    lower = np.array([142, 69, 14])  # -- Lower range --
    upper = np.array([145, 74, 18])  # -- Upper range --
    mask = cv2.inRange(state, lowerb=lower, upperb=upper)

    mask = cv2.bitwise_not(mask)
    state = cv2.bitwise_or(state, np.zeros_like(state), mask=mask)

    return state


class World:
    def __init__(self, device='cpu'):
        self.device = device
        self.verts = OrderedDict()
        self.faces = OrderedDict()
        self.colors = OrderedDict()
        self.models = OrderedDict()
        self.meshes = OrderedDict()

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
        self.colors[name] = torch.cat((color.to(self.device), torch.ones(V, 1, device=self.device)), dim=1)

    def create_scene(self, hws, alphas, N):
        batch = []
        for i in range(N):
            scene = []
            for mesh_name in self.models:
                hw = hws[mesh_name]
                alpha = alphas[mesh_name]
                N, K, _ = hw.shape
                for k in range(K):
                    c = self.colors[mesh_name].clone()
                    c[..., 3] = alpha[i, k]

                    textures = TexturesVertex(verts_features=[c])
                    m = Meshes(
                        verts=[self.verts[mesh_name].clone()],
                        faces=[self.faces[mesh_name].clone()],
                        textures=textures
                    )

                    t = Translate(y=hw[i, k, 0], x=hw[i, k, 1], z=torch.zeros(1, device=self.device), device=str(self.device))
                    m = m.update_padded(t.transform_points(m.verts_padded()))
                    scene += [m]
            batch += [join_meshes_as_scene(scene)]
        batch = join_meshes_as_batch(batch)
        return batch


class Tracker(nn.Module):
    def __init__(self, device):
        super().__init__()

        k = 1
        distance, elevation, azimuth = 30, 0.0, 0
        self.R, self.T = look_at_view_transform(distance, elevation, azimuth, device=device)
        meta = LayerMetaData(input_shape=(3, 128, 128))
        self.tracker, _ = make_layers(['C:3', 32, 64, 64, 32, 16, 8],
                                      type='resnet', meta=meta, nonlinearity=nn.SELU(inplace=True))

        self.tracker = self.tracker.to(device)
        cameras = FoVOrthographicCameras(device=device,
                                         max_x=64.0, max_y=64.0,
                                         min_x=-64.0, min_y=-64.0,
                                         scale_xyz=((1, 1, 1),))

        background_color = torch.tensor([144 / 255, 71 / 255, 16 / 255], device=device)
        blend_params = BlendParams(background_color=background_color)

        raster_settings = RasterizationSettings(
            image_size=128,
            blur_radius=1e-9,
            faces_per_pixel=k*5,
        )
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=CustomFlatShader(
                device=str(device),
                cameras=cameras,
                blend_params=blend_params
            )
        )

        self.particle_config = OrderedDict({
            'green_paddle': k,
            'top_wall': k,
            'bottom_wall': k,
            'puck': k,
            'red_paddle': k
        })

    def forward(self, image):
        heatmap = self.tracker(image)
        hws, alphas = sample_particles_from_heatmap_2d(heatmap[:, 0:5], k=self.particle_config, device=device,
                                                       h_min=-64.0, h_max=64.0,
                                                       w_min=-64, w_max=64.0,
                                                       deterministic=True)
        background_heatmap = heatmap[:, 0]

        N = 8

        scene = world.create_scene(hws, alphas, N)
        render, alpha = self.renderer(meshes_world=scene, R=self.R, T=self.T, background_heatmap=background_heatmap)
        return render, heatmap, alpha, hws


class Plot:
    def __init__(self, particle_config):
        self.particle_config = particle_config
        self.fig = plt.figure(figsize=(12, 18))
        self.ncols, self.nrows = 5, 6
        grid = gridspec.GridSpec(ncols=self.ncols, nrows=self.nrows, figure=self.fig)
        self.axe = [self.fig.add_subplot(self.nrows, self.ncols, i + 1) for i in range(self.ncols * (self.nrows-1))]
        self.loss_subplot = self.fig.add_subplot(grid[self.nrows-1, 0:self.ncols])
        self.loss = deque(maxlen=2000)

    def update_input(self, image):
        for i, name in enumerate(self.particle_config):
            i = i + self.ncols * 0
            self.axe[i].clear()
            self.axe[i].imshow(image.detach().cpu(), interpolation='nearest')
            self.axe[i].set_title(name)

    def update_heatmap(self, heatmap):
        for i, name in enumerate(self.particle_config):
            image = heatmap[i]
            i = i + self.ncols * 1
            self.axe[i].clear()
            self.axe[i].imshow(image.detach().cpu(), cmap='viridis', interpolation='nearest')

    def update_heatmap_grad(self, heatmap):
        for i, name in enumerate(self.particle_config):
            image = heatmap[i]
            i = i + self.ncols * 2
            self.axe[i].clear()
            self.axe[i].imshow(image.detach().cpu(), cmap='viridis', interpolation='nearest')

    def update_hws(self, hws):
        for i, (name, hw) in enumerate(hws.items()):
            i = i + self.ncols * 3
            self.axe[i].clear()
            self.axe[i].set_ylim(-64.0, 64.0)
            self.axe[i].set_xlim(-64.0, 64.0)
            self.axe[i].scatter(hw[0, :, 1].cpu(), hw[0, :, 0].cpu())

    def update_render(self, render):
        for i in range(self.ncols):
            i = i + self.ncols * 4
            self.axe[i].clear()
            self.axe[i].imshow(render.detach().cpu(), interpolation='nearest')

    def update_loss(self, loss):
        self.loss.append(loss.item())
        self.loss_subplot.clear()
        self.loss_subplot.plot(self.loss)
        print(loss.item())

    def show(self, pause):
        plt.pause(pause)


if __name__ == '__main__':

    """" generate dataset """

    def policy(state):
        """ a random policy to generate actions in Pong"""
        return randint(2, 3)

    pongdir = 'data/Pong_1'

    # if not Path(pongdir).exists():
    env = gym.make('Pong-v0')
    env = ClipState2D(env, 0, 24, 210 - 24, 160)
    env = ApplyFunc(env, remove_background)
    env = Resize2D(env, h=128, w=128)
    run = EnvRunner(env)
    run.attach_observer('image_cap', PngCapture(pongdir + '/screens', skip_first_n=30))
    run.episode(policy, render=False)

    train_dataset = torchvision.datasets.ImageFolder(
        root=pongdir,
        transform=torchvision.transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )

    """ load meshes from ply files into the world """
    world = World(device)

    for i, file in enumerate(Path('./data/meshes').iterdir()):
        name = str(file.stem)
        vert, normal, st, color, face = load_blender_ply_mesh(file)
        world.add_mesh(name, vert, face, color)

    tracker = Tracker(device=device)

    """ vizualization """
    heatmap_plot = Plot(tracker.particle_config)

    optimizer = torch.optim.Adam(tracker.parameters(), lr=1e-4)

    for epoch in range(20000):
        for batch, _ in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            render, heatmap, alpha, hws = tracker(batch)
            heatmap.retain_grad()
            # alpha = alpha.clone()
            # alpha[:, :, :, 0] = 0.0
            # loss_mask = alpha.max(dim=3, keepdim=True)[0]
            # loss_mask = loss_mask.detach()
            loss = (((batch.permute(0, 2, 3, 1) - render[..., 0:3]) ** 2)).mean()
            loss.backward()
            optimizer.step()
            heatmap_plot.update_loss(loss)
            heatmap_plot.update_input(batch.permute(0, 2, 3, 1)[0])
            heatmap_plot.update_heatmap(heatmap[0])
            heatmap_plot.update_heatmap_grad(heatmap.grad[0])
            heatmap_plot.update_render(render[0])
            heatmap_plot.update_hws(hws)
            # heatmap_plot.update_silhouette(loss_mask)
            heatmap_plot.show(0.05)
