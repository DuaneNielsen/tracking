from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from collections import OrderedDict
from utils import BoundingBoxes
from matplotlib import patches
import torch


class Buffer:
    def __init__(self, name, label_index, verts, faces, colors):
        self.name = name
        self.label_index = label_index
        self.verts = verts
        self.faces = faces
        self.colors = colors


class Model:
    def __init__(self, object):
        self.object = object
        self.mesh = None
        self.pose = None

    def update(self, pose):
        textures = TexturesVertex(verts_features=[self.object.colors.clone()])
        self.mesh = Meshes(
            verts=[self.object.verts.clone()],
            faces=[self.object.faces.clone()],
            textures=textures
        )
        self.pose = pose
        v = self.pose.transform_points(self.mesh.verts_padded())
        self.mesh = self.mesh.update_padded(v)


class Scene:
    def __init__(self):
        self.models = OrderedDict()

    def add_model(self, name, model):
        self.models[name] = model

    def update(self, poses):
        assert len(self.models) == len(poses)
        for i, (_, model) in enumerate(self.models.items()):
            model.update(poses[i])

    @property
    def mesh(self):
        return join_meshes_as_scene([model.mesh for _, model in self.models.items()])


class Pos:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def as_tuple(self):
        return self.x, self.y


class SimpleScreenBoundingBox:
    def __init__(self, mesh, camera, screen_size):
        """

           (x, y) = ([0, 0],[1, 1]) ||||||| (x, y) = ([1, 0], [1, 1])
           ||                                                      ||
           ||                                                      ||
           ||                                                      ||
           (x, y) = ([0, 0], [0, 1]) ||||||| (x, y) = ([1, 0], [0, 1])

        :param mesh: meshes to compute bounding boxes for
        :param camera: camera to project bounding boxes through
        :param screen_size: (h, w)
        """

        world_box = mesh.get_bounding_boxes()
        assert world_box.size(0) == 1
        box_verts = world_box[:, :, :].permute(0, 2, 1)
        self.box = camera.transform_points_screen(box_verts, torch.tensor([list(screen_size)]))[0]

    def top_left_bottom_right(self):
        return torch.cat((self.box[0, 0], self.box[1, 1], self.box[1, 0], self.box[0, 1]), dim=1)

    @property
    def bottom_left(self):
        """
        :return: tuple (x, y) co-ord
        """
        return Pos(self.box[0, 0], self.box[0, 1])

    @property
    def top_left(self):
        """
        :return: pos (x, y) co-ord
        """
        return Pos(self.box[0, 0], self.box[1, 1])

    @property
    def top_right(self):
        """
        :return: tuple (x, y) co-ord
        """
        return Pos(self.box[1, 0], self.box[1, 1])

    @property
    def bottom_right(self):
        """
        :return: tuple (x, y) co-ord
        """
        return Pos(self.box[1, 0], self.box[0, 1])

    @property
    def height(self):
        return self.box[1, 1] - self.box[0, 1]

    @property
    def width(self):
        return self.box[1, 0] - self.box[0, 0]

    def get_patch(self, linewidth=1, edgecolor='r', margin=(0, 0)):
        return patches.Rectangle((self.bottom_left.x - margin[0], self.bottom_left.y + margin[1]),
                                 height=self.height-margin[0]*2, width=self.width + margin[1]*2,
                                 linewidth=linewidth, edgecolor=edgecolor, facecolor='none')

    def save(self, file):
        pass

    @staticmethod
    def load(self):
        return BoundingBoxes()


class World:
    def __init__(self, device='cpu'):
        self.device = device
        self.objects = OrderedDict()
        self.object_index = []
        self.object_cursor = 0
        self.models = OrderedDict()
        self.meshes = OrderedDict()
        self.scenes = []

    def add_mesh(self, name, vert, face, color):
        """

        :param name: unique label for the mesh
        :param vert: V, 3 vert list
        :param face: F, 3 LongTensor
        :param color: V, 3 vertex colors
        """
        b = Buffer(name, self.object_cursor, vert.to(self.device), face.to(self.device), color.to(self.device))
        self.objects[name] = b
        self.object_index += [b]
        self.object_cursor += 1

    def create_scenes(self, scene_spec):
        """
        Create a scene from a scene spec
        :param scene_spec:[ [{ 'model_name' : 'object_name' }, ...], ... ] where model_name is a unique id in that
        scene, and 'object_name' is the name of an added mesh
        :return:
        """
        for s in scene_spec:
            scene = Scene()
            for model_name, object_name in s.items():
                scene.add_model(model_name, Model(self.objects[object_name]))
            self.scenes += [scene]

    def update_scenes(self, transforms):
        """
        Update the positions of all meshes in all scenes
        :param transforms: [[pytorch3d.Transform, ...], ...]
        :return: the scenes
        """
        assert len(self.scenes) == len(transforms)
        for scene, t in zip(self.scenes, transforms):
            scene.update(t)

    def batch(self):
        return join_meshes_as_batch([scene.mesh for scene in self.scenes])

    def labels(self):
        return [[m.object.label_index for _, m in s.models.items()] for s in self.scenes]

    def bounding_boxes(self, camera, screen_size):
        return [[SimpleScreenBoundingBox(m.mesh, camera, screen_size) for _, m in s.models.items()] for s in self.scenes]