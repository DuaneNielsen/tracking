from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pylab as plt
import gym

from pytorch3d.renderer import \
    (TexturesVertex, MeshRenderer, MeshRasterizer, FoVOrthographicCameras, RasterizationSettings,
     look_at_view_transform)
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from pytorch3d.transforms import Translate

from utils import load_blender_stl_mesh
from renderer import CustomFlatShader
from env.runner import EnvRunner, PngCapture
from env.wrappers import ClipState2D
from random import randint
import torch.utils.data
import torchvision.datasets


device = 'cuda' if torch.cuda.is_available() else 'cpu'


""" generate dataset """

pongdir = 'data/Pong_1'

def policy(state):
    """ a random policy to generate actions in Pong"""
    return randint(2, 3)


if not Path(pongdir).exists():
    env = gym.make('Pong-v0')
    env = ClipState2D(env, 0, 24, 210-24, 160)
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

cameras = FoVOrthographicCameras(device=device,
                                 max_x=80.0, max_y=93.0,
                                 min_x=-80.0, min_y=-93.0,
                                 scale_xyz=((1, 1, 1),))

raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=1e-9,
    faces_per_pixel=10,
)

custom_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=CustomFlatShader(
        device=str(device),
        cameras=cameras)
)

distance, elevation, azimuth = 30, 0.0, 0
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)


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
                c[..., 3] = c[..., 3] * alpha[i, k]
                textures = TexturesVertex(verts_features=[c])
                m = Meshes(
                    verts=[verts[mesh_name].clone()],
                    faces=[faces[mesh_name].clone()],
                    textures=textures
                )
                #m = meshes[mesh_name].clone().detach().to(device)
                t = Translate(y=hw[i, k, 0], x=hw[i, k, 1], z=torch.zeros(1, device=device), device=str(device))
                m = m.update_padded(t.transform_points(m.verts_padded()))
                scene += [m]
        batch += [join_meshes_as_scene(scene)]
    batch = join_meshes_as_batch(batch)
    return batch


# dummy function here N, K, D
hws = OrderedDict()
alphas = OrderedDict()

N = 8

# N, K co-ords and alphas
for name in models:
    hws[name] = torch.rand(N, 2, 2, device=device) * 10.0
    alphas[name] = torch.ones(N, 2, device=device)

scene = create_scene(hws, alphas, N)

batch = custom_renderer(meshes_world=scene, R=R, T=T)

fig = plt.figure(figsize=(16, 12))
axe = [fig.add_subplot(2, 4, i+1) for i in range(N)]

for i, image in enumerate(batch):
    axe[i].imshow(image.cpu())
plt.show()

