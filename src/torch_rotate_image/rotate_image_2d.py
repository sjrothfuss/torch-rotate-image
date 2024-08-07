import torch
import einops
from typing import Sequence
from torch_grid_utils import coordinate_grid
from torch_image_lerp import sample_image_2d


def rotate_image_2d(image: torch.Tensor, angles: float | torch.Tensor) -> torch.Tensor:
    """Rotate a 2D image by a given angle or angles.

    Parameters
    ----------
    image: torch.Tensor
        2D image to rotate. Shape should be `(h, w)`.
    angles: float | torch.Tensor
        Float angle or `(..., )` array of float angles in degrees by
        which to rotate the image.

    Returns
    -------
    rotated_image: torch.Tensor
        Rotated image. Shape is `(..., h, w)`.

    """
    if not isinstance(angles, torch.Tensor):
        angles = torch.tensor([angles], device=image.device)
    h, w = image.shape[-2:]
    # center = torch.tensor([h // 2, w // 2], device=image.device) # Alternative center
    center = _get_dft_center(image_shape=(h, w), device=image.device)

    coords = coordinate_grid(
        image_shape=(h, w),
        center=center,
        device=image.device,
    )  # (h, w, 2)
    coords = einops.rearrange(coords, "h w yx -> h w yx 1")

    rotation_matrices = _rotation_matrix_from_angles(angles)  # (..., 2, 2)
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 1 i j")

    rotated_coords = rotation_matrices @ coords  # (..., h, w, 2, 1)
    rotated_coords = einops.rearrange(rotated_coords, "... h w yx 1 -> ... h w yx")
    rotated_coords += center
    rotated_images = sample_image_2d(
        image=image,
        coordinates=rotated_coords,
    )  # (..., h, w)

    return rotated_images


def _rotation_matrix_from_angles(angles: torch.Tensor) -> torch.Tensor:
    """Get a 2D rotation matrix from angles in degrees.

    Parameters
    ----------
    angles: torch.Tensor
        Angles in degrees. Shape should be `(..., )`.

    Returns
    -------
    rotation_matrices: torch.Tensor
        Rotation matrices of shape `(..., 2, 2)`.

    """
    rotation_matrices = torch.empty(
        angles.shape + (2, 2),
        device=angles.device,
        dtype=angles.dtype,
    )
    angles_rad = torch.deg2rad(angles)
    cos = torch.cos(angles_rad)
    sin = torch.sin(angles_rad)
    rotation_matrices[..., 0, 0] = cos
    rotation_matrices[..., 0, 1] = -sin
    rotation_matrices[..., 1, 0] = sin
    rotation_matrices[..., 1, 1] = cos
    return rotation_matrices


def _get_dft_center(
    image_shape: tuple[int, ...],
    device: torch.device | None = None,
    rfft: bool = True,
    fftshifted: bool = True,
) -> torch.LongTensor:
    """Return the position of the center in an fftshifted DFT for a
    given input shape.

    This function makes explicit our convention used for selecting image
    centers.

    """
    fft_center = torch.zeros(size=(len(image_shape),), device=device)
    image_shape = torch.as_tensor(image_shape).float()
    if rfft is True:
        image_shape = torch.tensor(_get_rfft_shape(image_shape), device=device)
    if fftshifted is True:
        fft_center = torch.div(image_shape, 2, rounding_mode="floor")
    if rfft is True:
        fft_center[-1] = 0
    return fft_center.long()


def _get_rfft_shape(input_shape: Sequence[int]) -> tuple[int]:
    """Get the output shape of an rfft on an input of input_shape."""
    rfft_shape = list(input_shape)
    rfft_shape[-1] = int((rfft_shape[-1] / 2) + 1)
    return tuple(rfft_shape)
