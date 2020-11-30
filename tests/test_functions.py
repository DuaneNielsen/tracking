import torch

from collections import OrderedDict
from filter import axis, sample_particles_from_heatmap_2d

channel = torch.tensor([
    [
        [1, 0, 0, 2],
        [0, 0, 0, 0],
        [3, 0, 0, 4]
    ],
    [
        [0, 4, 0, 3],
        [0, 0, 0, 0],
        [1, 0, 0, 2]
    ]
])


def test_top_k():
    N, H, W = channel.shape
    h_i, w_i = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.long),
                              torch.linspace(0, W - 1, W, dtype=torch.long))
    h_i, w_i = h_i.flatten(), w_i.flatten()
    s, i = torch.topk(channel.flatten(1), 3, dim=1, sorted=True)
    alpha = torch.gather(channel.flatten(1), dim=1, index=i)
    h = h_i[i]
    w = w_i[i]

    expected_alpha = torch.tensor([
        [4, 3, 2],
        [4, 3, 2]
    ])
    assert torch.allclose(expected_alpha, alpha)
    expected_h = torch.tensor([
        [2, 2, 0],
        [0, 0, 2],
    ])
    assert torch.allclose(expected_h, h)
    expected_w = torch.tensor([
        [3, 0, 3],
        [1, 3, 3]
    ])
    assert torch.allclose(expected_w, w)


def test_sample_top_k():
    N, H, W = channel.shape
    h_i, w_i = torch.meshgrid(torch.linspace(0, H - 1, H, dtype=torch.long),
                              torch.linspace(0, W - 1, W, dtype=torch.long))
    h_i, w_i = h_i.flatten(), w_i.flatten()
    probs = torch.sigmoid(channel.float())
    with torch.no_grad():
        p = probs - torch.rand_like(probs)
        s, i = torch.topk(p.flatten(1), 5, dim=1, sorted=True)
    alpha = torch.gather(probs.flatten(1), dim=1, index=i)
    h = h_i[i]
    w = w_i[i]

    print('')
    print(probs)
    # print(alpha)
    # print(h)
    # print(w)

    mask = torch.zeros_like(probs).flatten(1)
    mask = mask.scatter(dim=1, index=i, src=alpha)
    mask = mask.reshape(N, H, W)

    print(mask)


def test_axis():
    expected_axis = torch.tensor([-4 / 5, -2 / 5, 0., 2 / 5, 4 / 5])
    assert torch.allclose(axis(-1.0, 1.0, 5), expected_axis)
    expected_reversed_axis = torch.tensor([4 / 5, 2 / 5, 0., -2 / 5, -4 / 5])
    assert torch.allclose(axis(min=-1.0, max=1.0, n=5, reversed=True), expected_reversed_axis)


def test_co_ordinates():
    channel = torch.tensor([
        [
            [1, 0, 0, 2],
            [0, 0, 0, 0],
            [3, 0, 0, 4]
        ],
        [
            [0, 4, 0, 3],
            [0, 0, 0, 0],
            [1, 0, 0, 2]
        ]
    ])

    hm = channel.unsqueeze(0)
    print(hm.shape)

    hw, alpha = sample_particles_from_heatmap_2d(hm, OrderedDict(first=4, second=2),
                                                 deterministic=True,
                                                 sorted=True,
                                                 h_min=-1.0,
                                                 w_min=-1.0)
    print('')
    assert len(hw) == 2

    N, K, D = hw[0].shape
    assert N == 1
    assert K == 4
    assert D == 2

    print(hw[0][0, 0], alpha[0][0, 0])
    print(hw[0][0, 1], alpha[0][0, 1])
    print(hw[0][0, 2], alpha[0][0, 2])
    print(hw[0][0, 3], alpha[0][0, 3])

    N, K, D = hw[1].shape
    assert N == 1
    assert K == 2
    assert D == 2


def test_sample_top_k_meshes():
    channel = torch.randn(2, 3, 4, 5)
    particles_per_mesh = torch.LongTensor([2, 3, 4])
    heatmap = torch.sigmoid(channel.float())
    hw, alpha = sample_particles_from_heatmap_2d(heatmap, OrderedDict(first=2, second=3, third=4))

    print([m[0, 0, :] for m in hw])
