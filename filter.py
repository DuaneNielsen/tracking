import torch
from collections import OrderedDict


def sample_top_k(probs, k, deterministic=False, sorted=False):
    """
    Samples cells from a batch of H, W heatmaps, more likely to sample high values
    heatmaps must be normalized between zero-> one
    :param heatmap: N, H, W, normalized between 0 and 1
    :param k: number of cells to sample
    :param deterministic: if True, doesn't sample, just returns the top k
    :return: indicies of sampled N, k
    """
    with torch.no_grad():
        p = probs - torch.rand_like(probs) if not deterministic else probs
        s, i = torch.topk(p.flatten(1), k, dim=1, sorted=sorted)
    return i


def axis(min, max, n, dtype=torch.float, reversed=False, device='cpu'):
    """

    :param n: number of intervals/cells/pixels on axis
    :param min: minimum co-ordinate on axis (ie -1 or 0)
    :param max: maximum co-ordinate on axis (1.0)
    :param dtype: return type
    :return: a tensor of length n containing the centered co-ordinate of each axis
    ie: the positions of x in the figure below

            -0.5      0       0.5
          |   x   |   x   |    x    |
    min -1.0                   max 1.0

    """

    l = max - min

    def p(i, l, n, min):
        return l/n * i + l/(2*n) + min

    start = p(0, l, n, min)
    end = p(n-1, l, n, min)
    if reversed:
        return torch.linspace(end, start, n, dtype=dtype, device=device)
    return torch.linspace(start, end, n, dtype=dtype, device=device)


def sample_particles_from_heatmap_2d(heatmap, k, h_min=0.0, w_min=0.0, h_max=1.0, w_max=1.0, deterministic=False, sorted=False, device='cpu'):
    """

    :param heatmap: N, M, H, W, a batch N of M heatmaps (one for each mesh) of H high and W wide, normalized between 0-1
    :param k: an OrderedDict with M entries, name: the label of the mesh, k, the number of particles to generate
    ie {'cup': 5, 'saucer', 3, 'spoon': 10 }, the order indicates which heatmap will be used
    :param h_min: minimum height co-ordinate
    :param w_min: minimum width co-ordinate
    :param h_max: maximum height co-ordinate
    :param w_max: maximum width co-ordinate
    :param deterministic: if True, always samples the k highest values
    :return: N K 2 - list of height width for each mesh, N K - list of alpha for each mesh
    height and width are floats containing an approximate value of the height or width, according to the axis function
    """
    N, M, H, W = heatmap.shape
    if M != len(k):
        raise Exception(f'Expected k to have {M} entries, one for each heatmap')
    h_axis = axis(h_min, h_max, H, dtype=torch.float, reversed=True, device=device)
    w_axis = axis(w_min, w_max, W, dtype=torch.float, device=device)
    h_i, w_i = torch.meshgrid(h_axis, w_axis)
    h_i, w_i = h_i.flatten(0), w_i.flatten(0)
    hw = OrderedDict()
    alpha = OrderedDict()
    for i, (label, k) in enumerate(k.items()):
        features = heatmap[:, i]
        index = sample_top_k(features, k, deterministic=deterministic, sorted=sorted)
        a = torch.gather(features.flatten(1), dim=1, index=index)
        h, w = h_i[index], w_i[index]
        hw[label] = torch.stack((h, w), dim=2)
        alpha[label] = a
    return hw, alpha