"""WIP module for 3D tensor rotation."""

import torch
import einops


def rotate_image_3d():
    raise NotImplementedError("Not implemented yet")


def _invert_3d_rotation_matrix(matrix: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(matrix)
