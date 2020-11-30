import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import utils
from pathlib import Path
from typing import Tuple
from collections import OrderedDict

# io utils
from pytorch3d.io import load_obj, load_ply

# datastructures
from pytorch3d.structures import Meshes, join_meshes_as_scene

# rendering components
from pytorch3d.renderer import (
    FoVPerspectiveCameras, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, HardPhongShader, PointLights, TexturesVertex, DirectionalLights,
    FoVOrthographicCameras, SoftPhongShader, SoftGouraudShader
)

from pytorch3d.transforms import Transform3d, Translate

from pytorch3d.renderer.materials import Materials
from pytorch3d.renderer.mesh.shading import flat_shading

# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


def load_blender_stl_mesh(file):
    verts, normals, sts, colors, faces = utils.load_ply(file)
    colors = colors[:, 0:3].to(torch.float) / 255.0
    return verts, normals, sts, colors, faces


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
    textures = TexturesVertex(verts_features=[color.to(device)])
    meshes[name] = Meshes(
        verts=[vert.to(device)],
        faces=[face.to(device)],
        textures=textures
    )


# dummy function here N, K, D
hw0 = torch.zeros(1, 5, 2, device=device)
alpha1 = torch.ones(1, 5, device=device)
hws = OrderedDict()
alphas = OrderedDict()

for name in models:
    hws[name] = hw0
    alphas[name] = alpha1

batch = []
scene = []

for i in range(1):
    for mesh_name in models:
        hw = hws[mesh_name]
        alpha = alphas[mesh_name]
        N, K, _ = hw.shape
        for k in range(K):
            m = meshes[mesh_name].clone().detach().to(device)
            t = Translate(x=hw[i, k, 0], y=hw[i, k, 1], z=torch.zeros(1, device=device), device=str(device))
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


def _apply_lighting(
    points, normals, lights, cameras, materials
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        points: torch tensor of shape (N, P, 3) or (P, 3).
        normals: torch tensor of shape (N, P, 3) or (P, 3)
        lights: instance of the Lights class.
        cameras: instance of the Cameras class.
        materials: instance of the Materials class.

    Returns:
        ambient_color: same shape as materials.ambient_color
        diffuse_color: same shape as the input points
        specular_color: same shape as the input points
    """
    light_diffuse = lights.diffuse(normals=normals, points=points)
    light_specular = lights.specular(
        normals=normals,
        points=points,
        camera_position=cameras.get_camera_center(),
        shininess=materials.shininess,
    )
    ambient_color = materials.ambient_color * lights.ambient_color
    diffuse_color = materials.diffuse_color * light_diffuse
    specular_color = materials.specular_color * light_specular
    if normals.dim() == 2 and points.dim() == 2:
        # If given packed inputs remove batch dim in output.
        return (
            ambient_color.squeeze(),
            diffuse_color.squeeze(),
            specular_color.squeeze(),
        )
    return ambient_color, diffuse_color, specular_color


def custom_shading(meshes, fragments, lights, cameras, materials, texels) -> torch.Tensor:
    """
    Apply per face shading. Use the average face position and the face normals
    to compute the ambient, diffuse and specular lighting. Apply the ambient
    and diffuse color to the pixel color and add the specular component to
    determine the final pixel color.

    Args:
        meshes: Batch of meshes
        fragments: Fragments named tuple with the outputs of rasterization
        lights: Lights class containing a batch of lights parameters
        cameras: Cameras class containing a batch of cameras parameters
        materials: Materials class containing a batch of material properties
        texels: texture per pixel of shape (N, H, W, K, 3)

    Returns:
        colors: (N, H, W, K, 3)
    """
    verts = meshes.verts_packed()  # (V, 3)
    faces = meshes.faces_packed()  # (F, 3)
    face_normals = meshes.faces_normals_packed()  # (V, 3)
    faces_verts = verts[faces]
    face_coords = faces_verts.mean(dim=-2)  # (F, 3, XYZ) mean xyz across verts

    # Replace empty pixels in pix_to_face with 0 in order to interpolate.
    mask = fragments.pix_to_face == -1
    pix_to_face = fragments.pix_to_face.clone()
    pix_to_face[mask] = 0

    N, H, W, K = pix_to_face.shape
    idx = pix_to_face.view(N * H * W * K, 1).expand(N * H * W * K, 3)

    # gather pixel coords
    pixel_coords = face_coords.gather(0, idx).view(N, H, W, K, 3)
    pixel_coords[mask] = 0.0
    # gather pixel normals
    pixel_normals = face_normals.gather(0, idx).view(N, H, W, K, 3)
    pixel_normals[mask] = 0.0

    return texels


def custom_rgb_blend(colors, fragments, blend_params) -> torch.Tensor:
    """
    Naive blending of top K faces to return an RGBA image
      - **RGB** - choose color of the closest point i.e. K=0
      - **A** - 1.0

    Args:
        colors: (N, H, W, K, 3) RGB color for each of the top K faces per pixel.
        fragments: the outputs of rasterization. From this we use
            - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
              of the faces (in the packed representation) which
              overlap each pixel in the image. This is used to
              determine the output shape.
        blend_params: BlendParams instance that contains a background_color
        field specifying the color for the background
    Returns:
        RGBA pixel_colors: (N, H, W, 4)
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device

    # Mask for the background.
    is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)

    if torch.is_tensor(blend_params.background_color):
        background_color = blend_params.background_color.to(device)
    else:
        background_color = colors.new_tensor(blend_params.background_color)  # (3)

    # Find out how much background_color needs to be expanded to be used for masked_scatter.
    num_background_pixels = is_background.sum()

    # Set background color.
    pixel_colors = colors[..., 0, :].masked_scatter(
        is_background[..., None],
        background_color[None, :].expand(num_background_pixels, -1),
    )  # (N, H, W, 3)

    # Concat with the alpha channel.
    alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


def minimal_shading(meshes, fragments, lights, cameras, materials, texels):
    return texels


def minimal_blend(colors, fragments, blend_params):
    """
    Takes the top color and returns it
    :param colors:
    :param fragments:
    :param blend_params:
    :return:
    """
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    pixel_colors = colors[..., 0, :]
    alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
    return torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


class CustomFlatShader(nn.Module):
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
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
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
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = minimal_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = minimal_blend(colors, fragments, blend_params)
        return images


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
    blur_radius=1e-6,
    faces_per_pixel=5,
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
