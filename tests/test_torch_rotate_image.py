import torch
from torch_rotate_image import rotate_image_2d


def test_rotate_image_2d_shape():
    image = torch.rand(28, 28)
    angles = torch.tensor([0.0, 90.0])
    rotated_image = rotate_image_2d(image, angles)
    assert rotated_image.shape == (2, 28, 28)


def test_rotate_image_2d_rotation():
    image = torch.zeros(28, 28)
    image[:14, :] = 1  # image is half black, half white
    angles = torch.tensor([180.0, ])
    rotated_image = rotate_image_2d(image, angles)
    assert (rotated_image[0] == image) and (rotated_image[1] == image.flip(0))
