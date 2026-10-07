#!/usr/bin/env python3
"""Camera-only path-following controller for the follower robot."""

from __future__ import annotations

from collections import deque
import math
from typing import Optional

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformException, TransformListener

from .geometry import transform_matrix
from .vision import detect_markers


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class FollowerNode(Node):
    """Track tag 49 while following its observed trail instead of cutting turns."""

    def __init__(self):
        super().__init__('follower_node')
        self.declare_parameter('desired_distance', 1.15)
        self.declare_parameter('minimum_distance', 0.75)
        self.declare_parameter('maximum_linear_speed', 0.40)
        self.declare_parameter('maximum_angular_speed', 0.90)
        self.declare_parameter('linear_gain', 0.75)
        self.declare_parameter('angular_gain', 1.8)
        self.declare_parameter('lost_timeout', 1.20)

        self.desired_distance = float(self.get_parameter('desired_distance').value)
        self.minimum_distance = float(self.get_parameter('minimum_distance').value)
        self.max_linear = float(self.get_parameter('maximum_linear_speed').value)
        self.max_angular = float(self.get_parameter('maximum_angular_speed').value)
        self.linear_gain = float(self.get_parameter('linear_gain').value)
        self.angular_gain = float(self.get_parameter('angular_gain').value)
        self.lost_timeout = float(self.get_parameter('lost_timeout').value)

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image,
            '/follower/camera/image_raw',
            self.on_image,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            '/follower/camera/camera_info',
            self.on_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            '/follower/odom',
            self.on_odom,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(Twist, '/follower/cmd_vel', 10)
        self.range_pub = self.create_publisher(Float32, '/follower/range', 10)
        self.path_pub = self.create_publisher(Path, '/follower/path', 10)

        self.last_detection_time = -1.0
        self.last_distance = float('inf')
        self.minimum_observed_distance = float('inf')
        self.maximum_observed_distance = 0.0
        self.last_camera_bearing = 0.0
        self.follower_position: Optional[np.ndarray] = None
        self.follower_yaw = 0.0
        self.leader_trail = deque(maxlen=160)
        self.path = Path()
        self.path.header.frame_id = 'follower/odom'
        self.last_path_position: Optional[np.ndarray] = None
        self.create_timer(0.05, self.control)
        self.create_timer(5.0, self.report_range_metrics)
        self.get_logger().info('Follower ready: tracking rear ArUco tag 49.')

    def on_info(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
            self.distortion = np.asarray(msg.d, dtype=float)
            self.get_logger().info('Received follower camera calibration.')

    def on_odom(self, msg: Odometry) -> None:
        self.follower_position = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y], dtype=float
        )
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.follower_yaw = math.atan2(siny, cosy)
        if (
            self.last_path_position is None
            or np.linalg.norm(self.follower_position - self.last_path_position) >= 0.08
        ):
            pose = PoseStamped()
            pose.header = msg.header
            pose.header.frame_id = 'follower/odom'
            pose.pose = msg.pose.pose
            self.path.header.stamp = msg.header.stamp
            self.path.poses.append(pose)
            if len(self.path.poses) > 1200:
                self.path.poses = self.path.poses[-1200:]
            self.last_path_position = self.follower_position.copy()
            self.path_pub.publish(self.path)

    def on_image(self, msg: Image) -> None:
        if self.camera_matrix is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Follower image conversion failed: {exc}')
            return
        markers = detect_markers(
            image, 0.12, self.camera_matrix, self.distortion
        )
        tag = next(
            (
                marker
                for marker in markers
                if marker.marker_id == 49 and marker.tvec is not None
            ),
            None,
        )
        if tag is None:
            return

        point_camera = np.asarray(tag.tvec, dtype=float).reshape(3)
        right, forward = float(point_camera[0]), float(point_camera[2])
        if forward <= 0.0:
            return
        measured_distance = math.hypot(right, forward)
        # Reject isolated pose-estimation spikes.
        if (
            math.isfinite(self.last_distance)
            and self.last_detection_time >= 0.0
            and abs(measured_distance - self.last_distance) > 1.2
        ):
            return
        self.last_distance = measured_distance
        self.minimum_observed_distance = min(
            self.minimum_observed_distance, measured_distance
        )
        self.maximum_observed_distance = max(
            self.maximum_observed_distance, measured_distance
        )
        self.range_pub.publish(Float32(data=float(measured_distance)))
        self.last_camera_bearing = math.atan2(right, forward)
        self.last_detection_time = self.get_clock().now().nanoseconds * 1.0e-9
        self._append_trail_point(msg, point_camera)

    def _append_trail_point(self, msg: Image, point_camera: np.ndarray) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                'follower/odom',
                msg.header.frame_id,
                Time.from_msg(msg.header.stamp),
                timeout=Duration(seconds=0.06),
            )
        except TransformException:
            try:
                transform = self.tf_buffer.lookup_transform(
                    'follower/odom',
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=0.03),
                )
            except TransformException:
                return
        homogeneous = np.ones(4, dtype=float)
        homogeneous[:3] = point_camera
        point = (transform_matrix(transform) @ homogeneous)[:2]
        if not self.leader_trail:
            self.leader_trail.append(point)
            return
        separation = float(np.linalg.norm(point - self.leader_trail[-1]))
        if 0.10 < separation < 1.0:
            self.leader_trail.append(point)

    def _trail_heading_error(self) -> Optional[float]:
        if self.follower_position is None:
            return None
        while len(self.leader_trail) > 1:
            distance = np.linalg.norm(
                self.leader_trail[0] - self.follower_position
            )
            if distance >= 0.42:
                break
            self.leader_trail.popleft()
        if not self.leader_trail:
            return None
        target = self.leader_trail[0]
        delta = target - self.follower_position
        if np.linalg.norm(delta) < 0.15:
            return None
        desired_yaw = math.atan2(float(delta[1]), float(delta[0]))
        return _wrap(desired_yaw - self.follower_yaw)

    def report_range_metrics(self) -> None:
        if not math.isfinite(self.minimum_observed_distance):
            return
        self.get_logger().info(
            'Follower tag range so far: '
            f'min={self.minimum_observed_distance:.2f} m, '
            f'max={self.maximum_observed_distance:.2f} m'
        )

    def control(self) -> None:
        command = Twist()
        now = self.get_clock().now().nanoseconds * 1.0e-9
        age = now - self.last_detection_time
        if self.last_detection_time < 0.0 or age > self.lost_timeout:
            # A short in-place recovery cannot drive into a wall. Stop completely
            # after 2 seconds rather than blindly advancing without the tag.
            if self.last_detection_time >= 0.0 and age < 2.0:
                direction = -1.0 if self.last_camera_bearing > 0.0 else 1.0
                command.angular.z = 0.16 * direction
            self.cmd_pub.publish(command)
            return

        heading_error = self._trail_heading_error()
        if heading_error is None:
            # Optical +x is right, while positive robot yaw is left.
            heading_error = -self.last_camera_bearing
        command.angular.z = _clamp(
            self.angular_gain * heading_error,
            -self.max_angular,
            self.max_angular,
        )

        if self.last_distance > self.minimum_distance and abs(heading_error) < 0.75:
            speed = self.linear_gain * (self.last_distance - self.desired_distance)
            if self.last_distance > 1.8:
                speed = max(speed, 0.16)
            command.linear.x = _clamp(speed, 0.0, self.max_linear)
        self.cmd_pub.publish(command)


def main():
    rclpy.init()
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
