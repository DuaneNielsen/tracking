from pytorch3d.io.ply_io import _load_ply_raw
from pytorch3d.io.utils import _check_faces_indices, _make_tensor, _open_file
import torch
import numpy as np


def load_ply(f):
    """
    Load the data from a .ply file.

    Example .ply file format:

    ply
    format ascii 1.0           { ascii/binary, format version number }
    comment made by Greg Turk  { comments keyword specified, like all lines }
    comment this file is a cube
    element vertex 8           { define "vertex" element, 8 of them in file }
    property float x           { vertex contains float "x" coordinate }
    property float y           { y coordinate is also a vertex property }
    property float z           { z coordinate, too }
    element face 6             { there are 6 "face" elements in the file }
    property list uchar int vertex_index { "vertex_indices" is a list of ints }
    end_header                 { delimits the end of the header }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.

    Returns:
        verts: FloatTensor of shape (V, 3).
        faces: LongTensor of vertex indices, shape (F, 3).
    """
    header, elements = _load_ply_raw(f)

    vertex = elements.get("vertex", None)
    if vertex is None:
        raise ValueError("The ply file has no vertex element.")

    face = elements.get("face", None)
    if face is None:
        raise ValueError("The ply file has no face element.")

    # if len(vertex) and (
    #     not isinstance(vertex, np.ndarray) or vertex.ndim != 2 or vertex.shape[1] != 3
    # ):
    #     raise ValueError("Invalid vertices in file.")
    vertex_pos = [vertex[0:3] for vertex in vertex]
    vertex_n = [vertex[3:6] for vertex in vertex]
    verts_st = [vertex[5:8] for vertex in vertex]
    vertex_rgba = [vertex[8:13] for vertex in vertex]

    verts = _make_tensor(vertex_pos, cols=3, dtype=torch.float32)
    verts_normal = _make_tensor(vertex_n, cols=3, dtype=torch.float32)
    verts_st = _make_tensor(verts_st, cols=2, dtype=torch.float32)
    verts_rgba = _make_tensor(vertex_rgba, cols=4, dtype=torch.uint8)

    face_head = next(head for head in header.elements if head.name == "face")
    if len(face_head.properties) != 1 or face_head.properties[0].list_size_type is None:
        raise ValueError("Unexpected form of faces data.")
    # face_head.properties[0].name is usually "vertex_index" or "vertex_indices"
    # but we don't need to enforce this.

    if not len(face):
        faces = torch.zeros(size=(0, 3), dtype=torch.int64)
    elif isinstance(face, np.ndarray) and face.ndim == 2:  # Homogeneous elements
        if face.shape[1] < 3:
            raise ValueError("Faces must have at least 3 vertices.")
        face_arrays = [face[:, [0, i + 1, i + 2]] for i in range(face.shape[1] - 2)]
        faces = torch.LongTensor(np.vstack(face_arrays).astype(np.long))
    else:
        face_list = []
        for face_item in face:
            if face_item.ndim != 1:
                raise ValueError("Bad face data.")
            if face_item.shape[0] < 3:
                raise ValueError("Faces must have at least 3 vertices.")
            for i in range(face_item.shape[0] - 2):
                face_list.append([face_item[0], face_item[i + 1], face_item[i + 2]])
        faces = _make_tensor(face_list, cols=3, dtype=torch.int64)

    _check_faces_indices(faces, max_index=verts.shape[0])
    return verts, verts_normal, verts_st, verts_rgba, faces

def load_blender_stl_mesh(file):
    verts, normals, sts, colors, faces = load_ply(file)
    colors = colors[:, 0:3].to(torch.float) / 255.0
    return verts, normals, sts, colors, faces

if __name__ == '__main__':

    verts, verts_nomal, verts_st, verts_rgba, faces = load_ply('data/green_paddle.ply')
    print(verts_rgba)


