import engine
import torch
import torch.nn as nn
from pytorch3d.transforms import Translate

from pytorch3d.renderer import FoVOrthographicCameras, TexturesVertex, look_at_view_transform, MeshRasterizer, \
    MeshRenderer, RasterizationSettings
from matplotlib import pyplot as plt
from matplotlib import patches

verts = torch.tensor([
    [-2, -1, 0],
    [2, -1, 0],
    [2, 1, 0],
    [-2, 1, 0]
], dtype=torch.float) * 4.0

faces = torch.LongTensor([
    [0, 1, 2], [0, 2, 3]
])

white = torch.ones_like(verts)
red = white * torch.tensor([1.0, 0.0, 0.0])
green = white * torch.tensor([0.0, 1.0, 0.0])
blue = white * torch.tensor([0.0, 0.0, 1.0])


class IdentityShader(nn.Module):
    """
    shader that simply returns the raw texels
    """

    def __init__(self):
        super().__init__()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        texels = meshes.sample_textures(fragments)
        return texels


def test_scene():
    world = engine.World()

    world.add_mesh('red_box', verts, faces, red)
    world.add_mesh('green_box', verts, faces, green)
    world.add_mesh('blue_box', verts, faces, blue)

    scene_spec = [
        {'red_box_0': 'red_box', 'green_box_0': 'green_box'},
        {'blue_box_0': 'blue_box', 'blue_box_1': 'blue_box'}
    ]

    world.create_scenes(scene_spec)

    poses = [
        [Translate(0, -30, 0), Translate(-10, -10, 0)],
        [Translate(40, 0, 0), Translate(-10, -10, 0)]
    ]

    world.update_scenes(poses)

    batch = world.batch()
    labels = world.labels()

    distance = 30
    elevation = 0.0
    azimuth = 0

    R, T = look_at_view_transform(distance, elevation, azimuth)
    cameras = FoVOrthographicCameras(max_x=64.0, max_y=64.0,
                                     min_x=-64.0, min_y=-64.0,
                                     scale_xyz=((1, 1, 1),),
                                     R=R, T=T)

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

    boxes = world.bounding_boxes(cameras, (128, 128))
    image = renderer(batch)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image[0, :, :, 0, :])
    for box in boxes[0]:
        ax.add_patch(box.get_patch())
    plt.show()