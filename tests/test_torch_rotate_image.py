"""Tests for the torch_rotate_image module."""

import torch
import matplotlib.pyplot as plt
from torch_rotate_image import rotate_image_2d
from torch_rotate_image.rotate_image_2d import _get_dft_center

TOLERANCE = 1e-6


def display_tensor(tensor: torch.Tensor, title: str = "") -> None:
    """Helper function to display a tensor as an image during debugging."""
    if tensor.ndim == 3:
        tensor = tensor.squeeze(0)
    plt.figure()
    plt.imshow(tensor.numpy(), cmap="viridis")
    plt.title(title)
    plt.clim(0, 1)
    plt.colorbar()
    plt.show()


def test_rotate_image_2d_shape() -> None:
    """Test that image shape is correct after rotation."""
    image = torch.rand(28, 28)
    angles = torch.tensor([0.0, 90.0])
    rotated_image = rotate_image_2d(image, angles)
    assert rotated_image.shape == (2, 28, 28)


def test_rotate_image_2d_flip() -> None:
    """To test that rotating an image 180 degrees is the same as flipping it."""
    image = torch.linspace(0.5, 1, steps=28).repeat(28, 1)
    flipped_image = image.flip(1)[:-1, :-1]  # mask out edge artifacts
    rotated_image = rotate_image_2d(image=image, angles=180.0)
    rotated_image = rotated_image[:, 1:, 1:]  # mask out edge artifacts
    assert torch.allclose(rotated_image, flipped_image, atol=TOLERANCE)


def test_rotate_image_2d_rotations() -> None:
    """Test that values are correct after rotating an image.

    Note: these values were set by running the function so it is a test
    for consistency between versions, not necessarily a ground truth.
    """
    image = torch.linspace(0.5, 1, steps=3).repeat(3, 1)
    angles = torch.tensor([0.0, 42.0])
    rotated_image = rotate_image_2d(image, angles)
    expected_tensor = torch.tensor(
        [
            [
                [0.50, 0.75, 1.00],
                [0.50, 0.75, 1.00],
                [0.50, 0.75, 1.00],
            ],
            [
                [0.0000000, 0.9172826, 0.0000000],
                [0.5642138, 0.7500000, 0.9357861],
                [0.0000000, 0.5827173, 0.0000000],
            ],
        ]
    )
    assert torch.allclose(rotated_image, expected_tensor, atol=TOLERANCE)


def test_rotate_image_2d_circular_symmetry() -> None:
    """Test that rotations by θ and by θ ± 360 are equivalent.

    This also tests rotation by negative angles.
    """
    image = torch.rand(20, 20)
    theta = 4.2
    assert torch.allclose(
        rotate_image_2d(image, theta),
        rotate_image_2d(image, theta + 360),
        atol=TOLERANCE,
    ) and torch.allclose(
        rotate_image_2d(image, theta),
        rotate_image_2d(image, theta - 360),
        atol=TOLERANCE,
    )


def test_get_dft_center() -> None:
    """Test that the center of the DFT is calculated correctly."""
    h, w = 10, 10
    center = _get_dft_center(image_shape=(h, w), rfft=False)
    assert torch.equal(center, torch.tensor((5, 5)))
