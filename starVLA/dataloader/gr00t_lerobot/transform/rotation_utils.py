# SPDX-FileCopyrightText: Copyright (c) 2025 starVLA community
# SPDX-License-Identifier: MIT

"""
JIT-optimized quaternion and rotation utilities.

Ported from GalaxeaVLA with compatibility for starVLA architecture.
"""

import torch
import torch.nn.functional as F


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first (w, i, j, k),
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1).clamp(min=1e-8)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Quaternions with real part first (w, i, j, k), as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = torch.sqrt(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        ).clamp(min=0.0)
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first (w, i, j, k),
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians for each matrix as a tensor of shape (...).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    def _index_from_letter(letter: str) -> int:
        if letter == "X":
            return 0
        if letter == "Y":
            return 1
        if letter == "Z":
            return 2
        raise ValueError("letter must be either X, Y or Z.")

    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2

    if tait_bryan:
        central_angle = torch.asin(
            (matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)).clamp(-1.0, 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0].clamp(-1.0, 1.0))

    def _angle_from_tan(axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool):
        i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
        if horizontal:
            i2, i1 = i1, i2
        even = (axis + other_axis) in ["XY", "YZ", "ZX"]
        if horizontal == even:
            return torch.atan2(data[..., i1], data[..., i2])
        if tait_bryan:
            return torch.atan2(-data[..., i2], data[..., i1])
        return torch.atan2(data[..., i2], -data[..., i1])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. to rotation matrix
    using Gram--Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        Batch of rotation matrices of size (*, 3, 3)

    Reference:
        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by dropping
    the last row.

    Args:
        matrix: Batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    Reference:
        Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def rotation_9d_to_matrix(d9: torch.Tensor) -> torch.Tensor:
    """
    Convert 9D rotation representation to rotation matrix.

    Args:
        d9: 9D rotation representation of size (..., 9)

    Returns:
        Batch of rotation matrices of size (..., 3, 3)
    """
    assert d9.shape[-1] == 9
    matrices = d9.reshape(-1, 3, 3).permute(0, 2, 1)
    u, s, vt = torch.linalg.svd(matrices)
    det = torch.det(torch.matmul(u, vt))
    vt[:, :, -1] *= det[:, None]
    rot_matrices = torch.matmul(u, vt)
    rot_matrices = rot_matrices.unflatten(0, d9.shape[:-1])
    return rot_matrices


def matrix_to_rotation_9d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 9D rotation representation.

    Args:
        matrix: Batch of rotation matrices of size (..., 3, 3)

    Returns:
        9D rotation representation of size (..., 9)
    """
    assert matrix.shape[-2:] == (3, 3)
    return matrix.transpose(-2, -1).flatten(-2, -1)


# ============================================================================
# JIT-compiled fused functions for speed
# ============================================================================

@torch.jit.script
def quaternion_conjugate_jit(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugate of quaternion in (i, j, k, w) format.

    Args:
        q: Quaternions as tensor of shape (..., 4) in (i, j, k, w) format.

    Returns:
        Conjugated quaternions.
    """
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


@torch.jit.script
def quaternion_multiply_jit(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions in (i, j, k, w) format.

    Args:
        q1: First quaternion (..., 4) in (i, j, k, w).
        q2: Second quaternion (..., 4) in (i, j, k, w).

    Returns:
        Product quaternion (..., 4) in (i, j, k, w).
    """
    i1, j1, k1, r1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    i2, j2, k2, r2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    r = r1 * r2 - i1 * i2 - j1 * j2 - k1 * k2
    i = r1 * i2 + i1 * r2 + j1 * k2 - k1 * j2
    j = r1 * j2 - i1 * k2 + j1 * r2 + k1 * i2
    k = r1 * k2 + i1 * j2 - j1 * i2 + k1 * r2

    return torch.stack([i, j, k, r], dim=-1)


@torch.jit.script
def quaternion_rotate_vector_jit(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate vector v by quaternion q in (i, j, k, w) format.

    Args:
        q: Quaternion (..., 4) in (i, j, k, w) format.
        v: Vector (..., 3).

    Returns:
        Rotated vector (..., 3).
    """
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]

    # t = 2 * cross(q.xyz, v)
    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)

    # v' = v + q.w * t + cross(q.xyz, t)
    rx = x + qw * tx + (qy * tz - qz * ty)
    ry = y + qw * ty + (qz * tx - qx * tz)
    rz = z + qw * tz + (qx * ty - qy * tx)

    return torch.stack([rx, ry, rz], dim=-1)


@torch.jit.script
def quaternion_to_rotation_6d_jit(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, i, j, k) to rotation_6d (first 2 rows of rotation matrix).

    Args:
        quaternion: Quaternion in (w, i, j, k) format of shape (..., 4).

    Returns:
        Rotation 6D representation of shape (..., 6).
    """
    r, i, j, k = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    two_s = 2.0 / (quaternion * quaternion + 1e-8).sum(-1)

    m00 = 1 - two_s * (j * j + k * k)
    m01 = two_s * (i * j - k * r)
    m02 = two_s * (i * k + j * r)
    m10 = two_s * (i * j + k * r)
    m11 = 1 - two_s * (i * i + k * k)
    m12 = two_s * (j * k - i * r)

    return torch.stack([m00, m01, m02, m10, m11, m12], dim=-1)


@torch.jit.script
def quaternion_to_rotation_9d_jit(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, i, j, k) to rotation_9d (all 9 elements, column-major).

    Args:
        quaternion: Quaternion in (w, i, j, k) format of shape (..., 4).

    Returns:
        Rotation 9D representation of shape (..., 9).
    """
    r, i, j, k = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    two_s = 2.0 / (quaternion * quaternion + 1e-8).sum(-1)

    # All 3 columns
    col0_0 = 1 - two_s * (j * j + k * k)
    col0_1 = two_s * (i * j + k * r)
    col0_2 = two_s * (i * k - j * r)

    col1_0 = two_s * (i * j - k * r)
    col1_1 = 1 - two_s * (i * i + k * k)
    col1_2 = two_s * (j * k + i * r)

    col2_0 = two_s * (i * k + j * r)
    col2_1 = two_s * (j * k - i * r)
    col2_2 = 1 - two_s * (i * i + j * j)

    return torch.stack(
        [col0_0, col0_1, col0_2, col1_0, col1_1, col1_2, col2_0, col2_1, col2_2], dim=-1
    )


@torch.jit.script
def rotation_6d_to_quaternion_jit(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation_6d to quaternion (w, i, j, k) via Gram-Schmidt.

    Args:
        d6: Rotation 6D representation of shape (..., 6).

    Returns:
        Quaternion in (w, i, j, k) format of shape (..., 4).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack into matrix
    matrix = torch.stack([b1, b2, b3], dim=-2)

    # Convert matrix to quaternion
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    trace = m00 + m11 + m22

    q_abs = torch.zeros(matrix.shape[:-2] + (4,), dtype=matrix.dtype, device=matrix.device)

    # Case 1: trace > 0
    mask0 = trace > 0
    s0 = torch.sqrt(trace[mask0] + 1.0) * 2
    q_abs[mask0, 0] = 0.25 * s0
    q_abs[mask0, 1] = (m21[mask0] - m12[mask0]) / s0
    q_abs[mask0, 2] = (m02[mask0] - m20[mask0]) / s0
    q_abs[mask0, 3] = (m10[mask0] - m01[mask0]) / s0

    # Case 2: m00 is largest
    mask1 = (~mask0) & (m00 > m11) & (m00 > m22)
    s1 = torch.sqrt(1.0 + m00[mask1] - m11[mask1] - m22[mask1]) * 2
    q_abs[mask1, 0] = (m21[mask1] - m12[mask1]) / s1
    q_abs[mask1, 1] = 0.25 * s1
    q_abs[mask1, 2] = (m01[mask1] + m10[mask1]) / s1
    q_abs[mask1, 3] = (m02[mask1] + m20[mask1]) / s1

    # Case 3: m11 is largest
    mask2 = (~mask0) & (~mask1) & (m11 > m22)
    s2 = torch.sqrt(1.0 + m11[mask2] - m00[mask2] - m22[mask2]) * 2
    q_abs[mask2, 0] = (m02[mask2] - m20[mask2]) / s2
    q_abs[mask2, 1] = (m01[mask2] + m10[mask2]) / s2
    q_abs[mask2, 2] = 0.25 * s2
    q_abs[mask2, 3] = (m12[mask2] + m21[mask2]) / s2

    # Case 4: m22 is largest
    mask3 = (~mask0) & (~mask1) & (~mask2)
    s3 = torch.sqrt(1.0 + m22[mask3] - m00[mask3] - m11[mask3]) * 2
    q_abs[mask3, 0] = (m10[mask3] - m01[mask3]) / s3
    q_abs[mask3, 1] = (m02[mask3] + m20[mask3]) / s3
    q_abs[mask3, 2] = (m12[mask3] + m21[mask3]) / s3
    q_abs[mask3, 3] = 0.25 * s3

    standardized = torch.where(q_abs[..., 0:1] < 0, -q_abs, q_abs)
    return standardized
