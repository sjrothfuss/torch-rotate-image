import torch
from torch_rotate_image import rotate_image_2d


def test_rotate_image_2d():
    image = torch.rand(28, 28)
    angles = torch.tensor([0.0, 90.0])
    rotated_image = rotate_image_2d(image, angles)
    assert rotated_image.shape == (2, 28, 28)


test_rotate_image_2d()
