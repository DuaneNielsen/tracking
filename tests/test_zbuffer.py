from pytorch3d.renderer import \
    (TexturesVertex, MeshRenderer, MeshRasterizer, FoVOrthographicCameras, RasterizationSettings,
     look_at_view_transform)
from pytorch3d.structures import Meshes, join_meshes_as_scene
import torch
import torch.nn as nn
import matplotlib.pylab as plt
from utils import load_blender_stl_mesh

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" a square mesh """

scale = 35.0

verts = torch.tensor([
    [-2, -1, 0],
    [2, -1, 0],
    [2, 1, 0],
    [-2, 1, 0]
], dtype=torch.float) * scale

faces = torch.LongTensor([
    [0, 1, 2], [0, 2, 3]
])

white = torch.ones_like(verts)
red = white * torch.tensor([1.0, 0.0, 0.0])
green = white * torch.tensor([0.0, 1.0, 0.0])
blue = white * torch.tensor([0.0, 0.0, 1.0])


def height(z):
    v = verts.clone()
    v[:, 2] = torch.full((4, ), fill_value=z)
    return v


meshes = Meshes(
    verts=[height(1.0), height(0.0), height(-1.0)],
    faces=[faces, faces, faces],
    textures=TexturesVertex([red, green, blue])
)


scene = join_meshes_as_scene(meshes)


class IdentityShader(nn.Module):
    """
    shader that simply returns the raw texels
    """

    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        return texels


cameras = FoVOrthographicCameras(device=device,
                                 max_x=80.0, max_y=93.0,
                                 min_x=-80.0, min_y=-93.0,
                                 scale_xyz=((1, 1, 1),))

raster_settings = RasterizationSettings(
    image_size=128,
    blur_radius=0,
    faces_per_pixel=6,
)




def render(renderer, scene):
    distance, elevation, azimuth = 30, 0.0, 0
    R, T = look_at_view_transform(distance, elevation, azimuth, device=device)
    return renderer(meshes_world=scene.to(device), R=R, T=T)


def plot_channels(image):
    fig = plt.figure(figsize=(10, 10))
    for i in range(image.size(-2)):
        panel = fig.add_subplot(3, 3, i+1)
        panel.imshow(image[..., i, :].squeeze().cpu())
    plt.grid(False)
    plt.show()


def test_zbuffer_render():
    zbuffer_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=IdentityShader()
    )

    plot_channels(render(zbuffer_renderer, scene))


def test_my_renderer():
    vert, normal, st, color, face = load_blender_stl_mesh('../data/meshes/background.ply')
    # V, _ = vert.shape
    # meshes = Meshes(
    #     verts=[vert.to(device)],
    #     faces=[face.to(device)],
    #     textures=TexturesVertex([torch.cat((color.to(device), torch.ones(V, 1, device=device)), dim=1)])
    # )

    # vert = torch.tensor([
    #     [-1, -1, 0],
    #     [1, -1, 0],
    #     [1, 1, 0],
    #     [-1, 1, 0]
    # ], dtype=torch.float) * 30

    x = 80
    y = 93

    vert = torch.tensor([
        [-x, -y, 0],
        [x, -y, 0],
        [x, y, 0],
        [-x, y, 0]
    ], dtype=torch.float)


    face = torch.LongTensor([
        [0, 1, 2], [0, 2, 3]
    ])

    meshes = Meshes(
        verts=[vert.to(device)],
        faces=[face.to(device)],
        textures=TexturesVertex([color.to(device)])
    )

    scene = join_meshes_as_scene(meshes)
    zbuffer_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=IdentityShader()
    )

    plot_channels(render(zbuffer_renderer, scene))