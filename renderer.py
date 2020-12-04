from typing import Union, List, Tuple

import torch
from pytorch3d.renderer import TexturesVertex, PointLights, Materials, BlendParams
from torch import nn as nn


class TexturesVertexAlpha(TexturesVertex):
    def __init__(
        self,
        verts_features: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
    ):
        """
        Batched texture representation where each vertex in a mesh
        has a D dimensional feature vector.

        Args:
            verts_features: list of (Vi, D) or (N, V, D) tensor giving a feature
                vector with artbitrary dimensions for each vertex.
        """
        if isinstance(verts_features, (tuple, list)):
            correct_shape = all(
                (torch.is_tensor(v) and v.ndim == 2) for v in verts_features
            )
            if not correct_shape:
                raise ValueError(
                    "Expected verts_features to be a list of tensors of shape (V, D)."
                )

            self._verts_features_list = verts_features
            self._verts_features_padded = None
            self.device = torch.device("cpu")

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            self._num_verts_per_mesh = [len(fv) for fv in verts_features]

            if self._N > 0:
                self.device = verts_features[0].device

        elif torch.is_tensor(verts_features):
            if verts_features.ndim != 4:
                msg = "Expected verts_features to be of shape (N, V, D); got %r"
                raise ValueError(msg % repr(verts_features.shape))
            self._verts_features_padded = verts_features
            self._verts_features_list = None
            self.device = verts_features.device

            # These values may be overridden when textures is
            # passed into the Meshes constructor. For more details
            # refer to the __init__ of Meshes.
            self._N = len(verts_features)
            max_F = verts_features.shape[1]
            self._num_verts_per_mesh = [max_F] * self._N
        else:
            raise ValueError("verts_features must be a tensor or list of tensors")

        # This is set inside the Meshes object when textures is
        # passed into the Meshes constructor. For more details
        # refer to the __init__ of Meshes.
        self.valid = torch.ones((self._N,), dtype=torch.bool, device=self.device)


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
    return colors[..., 0, :]
    #alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
    #torch.cat([pixel_colors, alpha], dim=-1)  # (N, H, W, 4)


def softmax_alpha_blend(colors, fragments, blend_params):
    N, H, W, K = fragments.pix_to_face.shape
    device = fragments.pix_to_face.device
    rgb = colors[..., 0:3]
    alpha = colors[..., 3]
    blend = torch.softmax(alpha, dim=-1)
    rgb = rgb * blend.unsqueeze(-1)
    rgb = rgb.sum(dim=3)
    alpha = torch.ones((N, H, W, 1), dtype=colors.dtype, device=device)
    return torch.cat([rgb, alpha], dim=-1)  # (N, H, W, 4)


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
        images = softmax_alpha_blend(colors, fragments, blend_params)
        return images