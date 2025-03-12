"""WIP 3D tensor rotation module."""

import torch
import einops


def rotate_image_3d():
    raise NotImplementedError("Not implemented yet")


def _invert_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(matrix)
