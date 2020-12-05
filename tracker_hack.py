import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

# io utils

# datastructures
from pytorch3d.structures import Meshes, join_meshes_as_scene

# rendering components
from pytorch3d.renderer import (
    look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    TexturesVertex, DirectionalLights,
    FoVOrthographicCameras
)

from pytorch3d.transforms import Translate

# Set the cuda device
from renderer import CustomFlatShader
from utils import load_blender_stl_mesh

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

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


# dummy function here N, K, D
hws = OrderedDict()
alphas = OrderedDict()

# N, K co-ords and alphas
for name in models:
    hws[name] = torch.zeros(1, 2, 2, device=device)
    alphas[name] = torch.ones(1, 2, device=device)

#hws['green_paddle'][0, 0, :] = torch.tensor([-30.0, 1.0])
#alphas['green_paddle'][0, 0] = torch.tensor([0.0])


batch = []
scene = []

for i in range(1):
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

teapot_mesh = join_meshes_as_scene(scene)

#for i, (name, particles) in particles_per_mesh:


#textures = TexturesVertex(verts_features=colors)

# meshes = Meshes(
#     verts=verts,
#     faces=faces,
#     textures=textures,
# )

# model_transforms = torch.zeros(len(models), 3)
# model_transforms[models['red_paddle']] = torch.tensor([0.0, 50.0, 0.0])
#
# t = Translate(model_transforms).to(device)
# scene = meshes.clone().detach().to(device)
# model_verts = scene.verts_padded()
# scene = scene.update_padded(t.transform_points(model_verts))
# teapot_mesh = join_meshes_as_scene(scene)


# Initialize a perspective camera.
cameras = FoVOrthographicCameras(device=device, max_x=80.0, max_y=93.0, min_x=-80.0, min_y=-93.0,
                                 scale_xyz=((1, 1, 1),))


lights = DirectionalLights(device=str(device),
                           direction=((0., 0., 1.),),
                           ambient_color=((0.9, 0.9, 0.9),),
                           diffuse_color=((0, 0, 0),),
                           specular_color=((0, 0, 0),),
                           )

raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=1e-9,
    faces_per_pixel=10,
)

background_color = torch.tensor([144 / 255, 71 / 255, 16 / 255], device=device)
blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=background_color)


custom_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=CustomFlatShader(
        device=str(device),
        cameras=cameras,
        blend_params=blend_params,
        lights=lights)
)

distance, elevation, azimuth = 30, 0.0, 0

plt.ion()
fig = plt.figure(figsize=(10, 10))
R, T = look_at_view_transform(distance, elevation, azimuth, device=device)

for i in range(200):
    phong_image_ref = custom_renderer(meshes_world=teapot_mesh, R=R, T=T)
    panel2 = fig.add_subplot(1, 1, 1)
    panel2.clear()
    panel2.set_facecolor('xkcd:salmon')
    panel2.imshow(phong_image_ref.squeeze().cpu())
    plt.grid(False)
    plt.show()
    plt.pause(0.05)
    distance -= 0.3


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer

        # image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 1).astype(np.float32))
        self.register_buffer('image_ref', image_ref[..., :3])

        self.camera_position = nn.Parameter(torch.tensor([3.0, 9.9, 2.5], device=device))

    def forward(self):
        R = look_at_rotation(self.camera_position[None, :], device=self.device)
        T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :, None])[:, :, 0]

        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)

        loss = torch.sum((image[..., :3] - self.image_ref) ** 2)
        return loss, image


# filename_outout = "./teapot_optimization_demo.gif"
#
# model = Model(meshes=teapot_mesh, renderer=custom_renderer, image_ref=phong_image_ref).to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
#
# plt.ion()
#
# _, image_init = model()
#
# fig = plt.figure(figsize=(10, 10))
# render_port = fig.add_subplot(1, 2, 1)
# render_port.imshow(image_init.detach().squeeze().cpu().numpy()[..., :3])
#
# target_image = fig.add_subplot(1, 2, 2)
# target_image.imshow(model.image_ref.squeeze().cpu().numpy())
#
# for i in tqdm(range(5000)):
#     optimizer.zero_grad()
#     loss, image = model()
#     loss.backward()
#     optimizer.step()
#
#     render_port = fig.add_subplot(1, 2, 1)
#     render_port.clear()
#     render_port.imshow(image.detach().squeeze().cpu().numpy()[..., :3])
#     plt.pause(0.1)
