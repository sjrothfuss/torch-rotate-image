import torch
from torch_rotate_image import rotate_image_2d
from torch_rotate_image.rotate_image_2d import _get_dft_center

TOLERANCE = 1e-6


def test_rotate_image_2d_shape():
    image = torch.rand(28, 28)
    angles = torch.tensor([0.0, 90.0])
    rotated_image = rotate_image_2d(image, angles)
    assert rotated_image.shape == (2, 28, 28)


def test_rotate_image_2d_rotation():
    image = torch.linspace(0.5, 1, steps=5).repeat(5, 1)
    rotated_image = rotate_image_2d(image=image, angles=180.0)
    expected_image = torch.tensor(
        [
            [
                [1.0000, 0.8750, 0.7500, 0.6250, 0.5000],
                [1.0000, 0.8750, 0.7500, 0.6250, 0.5000],
                [1.0000, 0.8750, 0.7500, 0.6250, 0.5000],
                [1.0000, 0.8750, 0.7500, 0.6250, 0.5000],
                [0.0000, 0.8750, 0.7500, 0.6250, 0.0000],
            ]
        ]
    )
    assert torch.allclose(rotated_image, expected_image, atol=TOLERANCE)


def test_get_dft_center():
    h, w = 10, 10
    center = _get_dft_center(image_shape=(h, w), device=torch.device("cpu"))
    assert center == (5, 5)
