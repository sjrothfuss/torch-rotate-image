import torch
import einops
from torch_grid_utils import coordinate_grid
from torch_image_lerp import sample_image_2d


def rotate_image_2d(image: torch.Tensor, angles: float | torch.Tensor) -> torch.Tensor:
    """Rotate a 2D image by a given angle or angles.

    Parameters
    ----------
    image: torch.Tensor
        2D image to rotate. Shape should be `(h, w)`.
    angles: float | torch.Tensor
        Angle or `(..., )` array of angles in degrees by which to rotate the image.

    Returns
    -------
    rotated_image: torch.Tensor
        Rotated image. Shape is `(..., h, w)`.
    """

    h, w = image.shape[-2:]
    grid = coordinate_grid(
        image_shape=(h, w), center=(h // 2, w // 2), device=image.device
    )  # (h, w, 2)
    # TODO make angles a Tensor
    rotation_matrices = _rotation_matrix_from_angles(angles)  # (..., 2, 2)
    # Prepare for broadcasting
    grid = einops.rearrange(grid, "h w yx -> h w yx 1")
    rotation_matrices = einops.rearrange(rotation_matrices, "... i j -> ... 1 1 i j")
    # (..., h, w, 2, 1) target shape for rotated output grid
    grid = rotation_matrices @ grid  # (..., h, w, 2, 1)
    grid = einops.rearrange(grid, "... h w yx 1 -> ... h w yx")
    rotated_images = sample_image_2d(image=image, coordinates=grid)  # (..., h, w)
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
