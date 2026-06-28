"""Quaternion operations, rotation conversions, and pose interpolation."""

import numpy as np
import torch


# --- NumPy utilities (used during data loading / preprocessing) ---


def euler_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles to a 3x3 rotation matrix.

    Args:
        roll: Rotation around X-axis in radians.
        pitch: Rotation around Y-axis in radians.
        yaw: Rotation around Z-axis in radians.

    Returns:
        (3, 3) rotation matrix as Rz @ Ry @ Rx.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def yaw_to_rotmat_np(yaw: float) -> np.ndarray:
    """Convert yaw angle to 3x3 rotation matrix (KITTI Y-axis convention).

    Args:
        yaw: Rotation around Y-axis in radians.

    Returns:
        (3, 3) float32 rotation matrix.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float32)


# --- Torch utilities (used during training / rendering) ---


def yaw_to_rotmat_torch(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw angle to 3x3 rotation matrix (KITTI Y-axis convention).

    Args:
        yaw: Scalar yaw tensor in radians.

    Returns:
        (3, 3) rotation matrix tensor.
    """
    cy = torch.cos(yaw)
    sy = torch.sin(yaw)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    return torch.stack([
        cy, zero, sy,
        zero, one, zero,
        -sy, zero, cy,
    ], dim=-1).reshape(3, 3)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Convert batch of quaternions to rotation matrices.

    Args:
        q: (N, 4) quaternions in [w, x, y, z] convention.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a quaternion.

    Args:
        R: (3, 3) rotation matrix tensor.

    Returns:
        (4,) quaternion in [w, x, y, z] convention.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return torch.stack([w, x, y, z])


def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Compute Hamilton product of two quaternion batches.

    Args:
        q1: (N, 4) quaternions in [w, x, y, z] convention.
        q2: (N, 4) quaternions in [w, x, y, z] convention.

    Returns:
        (N, 4) product quaternions.
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def slerp_quat(
    q0: torch.Tensor, q1: torch.Tensor, t: float
) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions.

    Args:
        q0: (4,) source quaternion [w, x, y, z].
        q1: (4,) target quaternion [w, x, y, z].
        t: Interpolation factor in [0, 1].

    Returns:
        (4,) interpolated quaternion.
    """
    dot = (q0 * q1).sum()
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = dot.clamp(-1, 1)
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    w0 = torch.sin((1 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    return w0 * q0 + w1 * q1


def interpolate_pose(
    R0: torch.Tensor,
    t0: torch.Tensor,
    R1: torch.Tensor,
    t1: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate between two rigid poses via slerp + lerp.

    Args:
        R0: (3, 3) source rotation matrix.
        t0: (3,) source translation.
        R1: (3, 3) target rotation matrix.
        t1: (3,) target translation.
        alpha: Interpolation factor in [0, 1].

    Returns:
        Tuple of (R_interp, t_interp).
    """
    t_interp = (1 - alpha) * t0 + alpha * t1
    q0 = rotmat_to_quat(R0)
    q1 = rotmat_to_quat(R1)
    q_interp = slerp_quat(q0, q1, alpha)
    q_interp = q_interp / q_interp.norm()
    R_interp = quat_to_rotmat(q_interp.unsqueeze(0))[0]
    return R_interp, t_interp
