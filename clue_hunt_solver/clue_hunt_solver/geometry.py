"""Small NumPy-only transform helpers shared by the ROS nodes."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import numpy as np


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized quaternion."""
    norm = x * x + y * y + z * z + w * w
    if norm < 1.0e-12:
        return np.eye(3, dtype=float)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def transform_matrix(transform) -> np.ndarray:
    """Convert a geometry_msgs Transform or TransformStamped to a 4x4 matrix."""
    value = getattr(transform, "transform", transform)
    q = value.rotation
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quaternion_matrix(q.x, q.y, q.z, q.w)
    matrix[:3, 3] = [value.translation.x, value.translation.y, value.translation.z]
    return matrix


def pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert OpenCV solvePnP output into a homogeneous matrix."""
    import cv2

    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rvec, dtype=float))[0]
    matrix[:3, 3] = np.asarray(tvec, dtype=float).reshape(3)
    return matrix


def yaw_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def project_pixel_to_ground(
    pixel: Tuple[float, float],
    camera_matrix: np.ndarray,
    map_from_camera: np.ndarray,
    ground_z: float = 0.0,
) -> Optional[np.ndarray]:
    """Project an optical-frame pixel ray onto a horizontal map-frame plane."""
    u, v = pixel
    ray_camera = np.linalg.inv(camera_matrix) @ np.array([u, v, 1.0])
    origin = map_from_camera[:3, 3]
    direction = map_from_camera[:3, :3] @ ray_camera
    if direction[2] >= -1.0e-4:
        return None
    distance = (ground_z - origin[2]) / direction[2]
    if distance <= 0.0:
        return None
    return origin + distance * direction


def relative_target(map_from_board: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply the challenge's board-frame ``REL a b`` command."""
    local = np.array([a, b, 0.0, 1.0], dtype=float)
    return (map_from_board @ local)[:3]


def median_xy(points: Iterable[np.ndarray]) -> Optional[np.ndarray]:
    values = [np.asarray(point, dtype=float)[:2] for point in points]
    if not values:
        return None
    return np.median(np.stack(values), axis=0)
