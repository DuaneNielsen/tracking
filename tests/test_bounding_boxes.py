from pytorch3d.renderer import FoVOrthographicCameras, TexturesVertex, look_at_view_transform, MeshRasterizer, \
    MeshRenderer, RasterizationSettings
from pytorch3d.structures import Meshes, join_meshes_as_scene
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib import patches
from utils import BoundingBoxes

class IdentityShader(nn.Module):
    """
    shader that simply returns the raw texels
    """

    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        return texels


def test_bounding_box():
    verts = torch.tensor([
        [-2, -1, 0],
        [2, -1, 0],
        [2, 1, 0],
        [-2, 1, 0]
    ], dtype=torch.float) * 20.0

    faces = torch.LongTensor([
        [0, 1, 2], [0, 2, 3]
    ])

    white = torch.ones_like(verts)
    red = white * torch.tensor([1.0, 0.0, 0.0])
    green = white * torch.tensor([0.0, 1.0, 0.0])
    blue = white * torch.tensor([0.0, 0.0, 1.0])

    meshes = Meshes(
        verts=[verts],
        faces=[faces],
        textures=TexturesVertex([blue])
    )

    distance = 30
    elevation = 0.0
    azimuth = 0

    R, T = look_at_view_transform(distance, elevation, azimuth)
    cameras = FoVOrthographicCameras(max_x=64.0, max_y=64.0,
                                     min_x=-64.0, min_y=-64.0,
                                     scale_xyz=((1, 1, 1),),
                                     R=R, T=T)

    box = meshes.get_bounding_boxes()
    box_as_verts = box[:, :, :].permute(0, 2, 1)
    box_screen = cameras.transform_points_screen(box_as_verts, torch.tensor([[128, 128]]))

    raster_settings = RasterizationSettings(
        image_size=128,
        blur_radius=0,
        faces_per_pixel=6,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings,
        ),
        shader=IdentityShader()
    )

    bb = BoundingBoxes(box_screen)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 10), constrained_layout=False)
    ax.imshow(renderer(meshes)[0, :, :, 0, :])

    boxes_rect = patches.Rectangle(bb.mpl_anchor(0), width=bb.width(0), height=bb.height(0), linewidth=4, edgecolor='r',
                                   facecolor='none')
    ax.add_patch(boxes_rect)
    plt.show()
