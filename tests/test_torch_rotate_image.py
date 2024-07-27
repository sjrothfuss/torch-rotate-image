import torch
from torch_rotate_image import rotate_image_2d

TOLERANCE = 1e-6


def test_rotate_image_2d_shape():
    image = torch.rand(28, 28)
    angles = torch.tensor([0.0, 90.0])
    rotated_image = rotate_image_2d(image, angles)
    assert rotated_image.shape == (2, 28, 28)


def test_rotate_image_2d_rotation():
    image = torch.linspace(0.5, 1, steps=28).repeat(28, 1)
    rotated_image = rotate_image_2d(image=image, angles=180.0)
    assert torch.allclose(rotated_image, image.flip(0), atol=TOLERANCE)
