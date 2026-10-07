#!/usr/bin/env python3
"""Autonomous leader for the Clue Chain Hunt challenge."""

from __future__ import annotations

from collections import deque
import math
from typing import Dict, Optional, Tuple

from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .clue import Clue, ClueError, validate_clue
from .geometry import (
    median_xy,
    pose_matrix,
    project_pixel_to_ground,
    relative_target,
    transform_matrix,
    yaw_quaternion,
)
from .vision import (
    decode_qr_beside_marker,
    decode_qr_codes,
    detect_markers,
    detect_pillars,
    pair_marker_with_qr,
)


class HuntNode(Node):
    """Read the authenticated clue chain and drive the leader with Nav2."""

    def __init__(self):
        super().__init__('hunt_node')
        self.declare_parameter('confirmation_frames', 2)
        self.declare_parameter('marker_length', 0.24)
        self.declare_parameter('search_angular_speed', 0.36)
        self.declare_parameter('board_standoff', 0.85)
        self.declare_parameter('process_period', 0.18)
        self.declare_parameter('navigation_timeout', 55.0)

        self.confirmation_frames = int(self.get_parameter('confirmation_frames').value)
        self.marker_length = float(self.get_parameter('marker_length').value)
        self.search_speed = float(self.get_parameter('search_angular_speed').value)
        self.board_standoff = float(self.get_parameter('board_standoff').value)
        self.process_period = float(self.get_parameter('process_period').value)
        self.navigation_timeout = float(
            self.get_parameter('navigation_timeout').value
        )

        self.bridge = CvBridge()
        self.camera_matrix: Optional[np.ndarray] = None
        self.distortion: Optional[np.ndarray] = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, '/scan', self.on_scan, qos_profile_sensor_data
        )
        map_qos = QoSProfile(depth=1)
        map_qos.reliability = ReliabilityPolicy.RELIABLE
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, '/map', self.on_map, map_qos)
        self.clue_pub = self.create_publisher(String, '/hunt/clues', 10)
        self.board_pub = self.create_publisher(String, '/hunt/boards', 10)
        self.treasure_pub = self.create_publisher(PoseStamped, '/hunt/treasure', 10)
        self.status_pub = self.create_publisher(String, '/leader/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/hunt/markers', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.navigator = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        self.previous_clue = 'START'
        self.expected_id = 1
        self.state = 'SEARCHING'
        self.candidate_text: Optional[str] = None
        self.candidate_count = 0
        self.accepted_texts = set()
        self.last_image_time = -1.0
        self.last_tf_warning_time = -10.0
        self.front_clearance = float('inf')

        self.pillar_samples: Dict[str, deque] = {
            colour: deque(maxlen=25) for colour in ('RED', 'GREEN', 'BLUE')
        }
        self.pending_clue: Optional[Tuple[Clue, np.ndarray]] = None
        self.pending_goal: Optional[Tuple[np.ndarray, bool, bool]] = None
        self.search_centre: Optional[np.ndarray] = None
        self.search_started = self.get_clock().now().nanoseconds * 1.0e-9
        self.search_waypoint_index = 0
        self.goal_handle = None
        self.goal_serial = 0
        self.goal_started = -1.0
        self.active_goal = None
        self.treasure_published = False
        self.visual_markers = {}
        self.hunt_started = self.get_clock().now()
        self.exploration_points = []
        self.exploration_visited = []

        self.create_timer(0.1, self.control_tick)
        self.create_timer(1.0, self.publish_status)
        self.publish_status()
        self.get_logger().info(
            'Leader ready: expecting board 1 with token for START.'
        )

    def on_info(self, msg: CameraInfo) -> None:
        if self.camera_matrix is None:
            self.camera_matrix = np.asarray(msg.k, dtype=float).reshape(3, 3)
            self.distortion = np.asarray(msg.d, dtype=float)
            self.get_logger().info('Received leader camera calibration.')

    def on_scan(self, msg: LaserScan) -> None:
        if not msg.ranges:
            return
        half_width = max(1, int(math.radians(18.0) / msg.angle_increment))
        centre = int((0.0 - msg.angle_min) / msg.angle_increment)
        values = []
        for index in range(centre - half_width, centre + half_width + 1):
            if 0 <= index < len(msg.ranges):
                value = msg.ranges[index]
                if math.isfinite(value):
                    values.append(value)
        self.front_clearance = min(values) if values else float('inf')

    def on_map(self, msg: OccupancyGrid) -> None:
        """Build generic free-space viewpoints directly from the saved map."""
        if self.exploration_points:
            return
        width, height = msg.info.width, msg.info.height
        if width == 0 or height == 0:
            return
        grid = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
        resolution = float(msg.info.resolution)
        clearance = max(1, int(math.ceil(0.48 / resolution)))
        stride = max(1, int(round(1.45 / resolution)))
        origin = msg.info.origin.position
        orientation = msg.info.origin.orientation
        siny = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cosy = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        yaw = math.atan2(siny, cosy)
        rotation = np.array([
            [math.cos(yaw), -math.sin(yaw)],
            [math.sin(yaw), math.cos(yaw)],
        ])
        points = []
        for row in range(clearance, height - clearance, stride):
            columns = range(clearance, width - clearance, stride)
            for column in columns:
                patch = grid[
                    row - clearance:row + clearance + 1,
                    column - clearance:column + clearance + 1,
                ]
                if patch.size == 0 or np.any(patch != 0):
                    continue
                local = np.array([
                    (column + 0.5) * resolution,
                    (row + 0.5) * resolution,
                ])
                points.append(
                    rotation @ local + np.array([origin.x, origin.y])
                )
        self.exploration_points = points
        self.get_logger().info(
            f'Generated {len(points)} map-derived exploration viewpoints.'
        )

    def _next_exploration_point(self) -> Optional[np.ndarray]:
        if not self.exploration_points:
            return None
        current = self._current_position()
        references = list(self.exploration_visited)
        if current is not None:
            references.append(current)
        candidates = []
        for point in self.exploration_points:
            if current is not None and np.linalg.norm(point - current) < 0.9:
                continue
            if any(np.linalg.norm(point - seen) < 0.9 for seen in self.exploration_visited):
                continue
            novelty = min(
                (np.linalg.norm(point - reference) for reference in references),
                default=0.0,
            )
            candidates.append((float(novelty), point))
        if not candidates:
            if self.exploration_visited:
                self.exploration_visited.clear()
                return self._next_exploration_point()
            return None
        point = max(candidates, key=lambda item: item[0])[1]
        self.exploration_visited.append(point)
        return point

    def _safe_viewpoint_near(self, desired: np.ndarray) -> Optional[np.ndarray]:
        """Snap a local search point onto collision-cleared map free space."""
        if not self.exploration_points:
            return np.asarray(desired, dtype=float)
        point = min(
            self.exploration_points,
            key=lambda candidate: np.linalg.norm(candidate - desired),
        )
        if np.linalg.norm(point - desired) > 1.05:
            return None
        return np.asarray(point, dtype=float)

    def on_image(self, msg: Image) -> None:
        if self.camera_matrix is None or self.state == 'DONE':
            return
        now_seconds = self.get_clock().now().nanoseconds * 1.0e-9
        if now_seconds - self.last_image_time < self.process_period:
            return
        self.last_image_time = now_seconds
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:  # cv_bridge uses several exception classes
            self.get_logger().error(f'Image conversion failed: {exc}')
            return

        map_from_camera = self._lookup_matrix('map', msg.header.frame_id, msg)
        if map_from_camera is not None:
            self._observe_pillars(image, map_from_camera)
            if self.pending_clue is not None:
                self._try_resolve_pending_clue()

        markers = detect_markers(
            image, self.marker_length, self.camera_matrix, self.distortion
        )
        expected_markers = [m for m in markers if m.marker_id == self.expected_id]
        if not expected_markers:
            return

        qr_codes = decode_qr_codes(image)
        for marker in expected_markers:
            if marker.rvec is None or marker.tvec is None:
                continue
            qr = pair_marker_with_qr(marker, qr_codes)
            if qr is None:
                qr = decode_qr_beside_marker(image, marker)
            if qr is None:
                continue
            try:
                clue = validate_clue(qr.text, self.expected_id, self.previous_clue)
            except ClueError as exc:
                self.get_logger().debug(f'Rejected QR {qr.text!r}: {exc}')
                continue
            if clue.board_id != marker.marker_id:
                continue

            map_from_board = self._map_from_marker(marker, msg)
            if map_from_board is None:
                continue
            if qr.text == self.candidate_text:
                self.candidate_count += 1
            else:
                self.candidate_text = qr.text
                self.candidate_count = 1
            if self.candidate_count >= self.confirmation_frames:
                self._accept_clue(clue, map_from_board)
                return

    def _lookup_matrix(self, target: str, source: str, msg=None) -> Optional[np.ndarray]:
        if not source:
            return None
        stamp = Time()
        if msg is not None:
            stamp = Time.from_msg(msg.header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                target, source, stamp, timeout=Duration(seconds=0.08)
            )
        except TransformException:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target, source, Time(), timeout=Duration(seconds=0.05)
                )
            except TransformException as exc:
                now = self.get_clock().now().nanoseconds * 1.0e-9
                if now - self.last_tf_warning_time > 3.0:
                    self.get_logger().warning(
                        f'Waiting for transform {target} <- {source}: {exc}'
                    )
                    self.last_tf_warning_time = now
                return None
        return transform_matrix(transform)

    def _map_from_marker(self, marker, msg: Image) -> Optional[np.ndarray]:
        map_from_camera = self._lookup_matrix('map', msg.header.frame_id, msg)
        if map_from_camera is None:
            return None
        return map_from_camera @ pose_matrix(marker.rvec, marker.tvec)

    def _observe_pillars(self, image: np.ndarray, map_from_camera: np.ndarray) -> None:
        for detection in detect_pillars(image):
            point = project_pixel_to_ground(
                detection.bottom_pixel, self.camera_matrix, map_from_camera
            )
            if point is None:
                continue
            camera_xy = map_from_camera[:2, 3]
            if np.linalg.norm(point[:2] - camera_xy) > 10.0:
                continue
            samples = self.pillar_samples[detection.colour]
            current = median_xy(samples)
            if current is not None and np.linalg.norm(point[:2] - current) > 0.9:
                continue
            samples.append(point[:2])
            if len(samples) >= 3:
                stable = median_xy(samples)
                if stable is not None:
                    self._publish_pillar_marker(detection.colour, stable)

    def _pillar_position(self, colour: str) -> Optional[np.ndarray]:
        samples = self.pillar_samples[colour]
        # A single projected mask can be a coloured texture or compression
        # artefact.  Require a short temporal consensus before navigating.
        if len(samples) < 3:
            return None
        return median_xy(samples)

    def _store_marker(
        self,
        namespace: str,
        marker_id: int,
        marker_type: int,
        position,
        scale,
        colour,
        text: str = '',
    ) -> None:
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x, marker.scale.y, marker.scale.z = map(float, scale)
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(
            float, colour
        )
        marker.text = text
        self.visual_markers[(namespace, marker_id)] = marker
        self.marker_pub.publish(
            MarkerArray(markers=list(self.visual_markers.values()))
        )

    def _publish_pillar_marker(self, colour: str, point: np.ndarray) -> None:
        rgb = {
            'RED': (0.90, 0.08, 0.08, 0.90),
            'GREEN': (0.08, 0.78, 0.16, 0.90),
            'BLUE': (0.08, 0.25, 0.92, 0.90),
        }[colour]
        marker_id = {'RED': 1, 'GREEN': 2, 'BLUE': 3}[colour]
        self._store_marker(
            'detected_pillars', marker_id, Marker.CYLINDER,
            (point[0], point[1], 0.50), (0.40, 0.40, 1.00), rgb,
        )
        self._store_marker(
            'labels', 100 + marker_id, Marker.TEXT_VIEW_FACING,
            (point[0], point[1], 1.25), (0.0, 0.0, 0.28), rgb, colour,
        )

    def _publish_board_marker(self, board_id: int, position: np.ndarray) -> None:
        self._store_marker(
            'accepted_boards', board_id, Marker.SPHERE,
            (position[0], position[1], 0.25), (0.28, 0.28, 0.28),
            (0.05, 0.72, 0.95, 0.95),
        )
        self._store_marker(
            'labels', 200 + board_id, Marker.TEXT_VIEW_FACING,
            (position[0], position[1], 0.65), (0.0, 0.0, 0.25),
            (0.05, 0.72, 0.95, 1.0), f'Board {board_id}',
        )

    def _accept_clue(self, clue: Clue, map_from_board: np.ndarray) -> None:
        if clue.text in self.accepted_texts:
            return
        self.accepted_texts.add(clue.text)
        self._stop_robot()
        self._cancel_navigation()
        self.state = 'READING'
        self.publish_status()

        self.clue_pub.publish(String(data=clue.text))
        board_position = map_from_board[:3, 3]
        self.board_pub.publish(
            String(data=f'{clue.board_id} {board_position[0]:.3f} {board_position[1]:.3f}')
        )
        self._publish_board_marker(clue.board_id, board_position)
        self.get_logger().info(
            f'Accepted board {clue.board_id}: {clue.text}; '
            f'position=({board_position[0]:.2f}, {board_position[1]:.2f})'
        )

        self.previous_clue = clue.text
        self.expected_id += 1
        self.candidate_text = None
        self.candidate_count = 0
        self.pending_clue = (clue, map_from_board)
        self.search_waypoint_index = 0
        self._try_resolve_pending_clue()

    def _try_resolve_pending_clue(self) -> None:
        if self.pending_clue is None:
            return
        clue, map_from_board = self.pending_clue
        target: Optional[np.ndarray] = None
        approach_board = True

        if clue.command == 'GOTO':
            target = np.array([clue.args[0], clue.args[1]], dtype=float)
        elif clue.command == 'REL':
            target = relative_target(
                map_from_board, float(clue.args[0]), float(clue.args[1])
            )[:2]
        elif clue.command == 'PILLAR':
            target = self._pillar_position(str(clue.args[0]))
        elif clue.command == 'BETWEEN':
            first = self._pillar_position(str(clue.args[0]))
            second = self._pillar_position(str(clue.args[1]))
            if first is not None and second is not None:
                fraction = float(clue.args[2])
                target = first + fraction * (second - first)
                approach_board = False

        if target is None:
            if self.state != 'LOCATING_PILLARS':
                self.search_started = (
                    self.get_clock().now().nanoseconds * 1.0e-9
                )
            self.state = 'LOCATING_PILLARS'
            self.publish_status()
            return

        self.pending_clue = None
        is_treasure = bool(clue.treasure)
        if is_treasure:
            approach_board = False
        self.search_centre = np.asarray(target, dtype=float)
        self.pending_goal = (self.search_centre, is_treasure, approach_board)
        self.state = 'WAITING_NAV'
        self.publish_status()

    def _current_position(self) -> Optional[np.ndarray]:
        matrix = self._lookup_matrix('map', 'base_footprint')
        return None if matrix is None else matrix[:2, 3]

    def _goal_pose(self, target: np.ndarray, approach: bool) -> PoseStamped:
        target = np.asarray(target, dtype=float)[:2]
        current = self._current_position()
        goal_xy = target.copy()
        if approach and current is not None:
            delta = target - current
            distance = float(np.linalg.norm(delta))
            if distance > self.board_standoff:
                goal_xy = target - self.board_standoff * delta / distance
        if current is None:
            yaw = 0.0
        elif np.linalg.norm(target - goal_xy) < 1.0e-3:
            # For exact/search goals, preserve the direction of travel instead
            # of forcing yaw=0 and making Nav2 spin at the destination.
            yaw = math.atan2(target[1] - current[1], target[0] - current[0])
        else:
            yaw = math.atan2(target[1] - goal_xy[1], target[0] - goal_xy[0])

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(goal_xy[0])
        pose.pose.position.y = float(goal_xy[1])
        qx, qy, qz, qw = yaw_quaternion(yaw)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        return pose

    def _send_navigation_goal(
        self, target: np.ndarray, is_treasure: bool, approach: bool
    ) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = self._goal_pose(target, approach)
        self.goal_serial += 1
        serial = self.goal_serial
        future = self.navigator.send_goal_async(goal)
        future.add_done_callback(
            lambda completed: self._on_goal_response(
                completed, serial, np.asarray(target), is_treasure
            )
        )
        self.goal_started = self.get_clock().now().nanoseconds * 1.0e-9
        self.active_goal = (np.asarray(target)[:2], is_treasure, approach)
        self.state = 'MOVING'
        self.publish_status()
        self.get_logger().info(
            f'Navigating toward ({target[0]:.2f}, {target[1]:.2f})'
        )

    def _on_goal_response(self, future, serial: int, target, is_treasure: bool) -> None:
        if serial != self.goal_serial:
            return
        try:
            handle = future.result()
        except Exception as exc:
            self.get_logger().error(f'Navigation goal failed: {exc}')
            self._begin_search()
            return
        if not handle.accepted:
            self.get_logger().warning('Nav2 rejected the goal; searching from here.')
            self._begin_search()
            return
        self.goal_handle = handle
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda completed: self._on_navigation_result(
                completed, serial, target, is_treasure
            )
        )

    def _on_navigation_result(self, future, serial: int, target, is_treasure: bool) -> None:
        if serial != self.goal_serial:
            return
        self.goal_handle = None
        self.active_goal = None
        try:
            status = future.result().status
        except Exception as exc:
            self.get_logger().error(f'Navigation result failed: {exc}')
            status = GoalStatus.STATUS_UNKNOWN
        if is_treasure:
            current = self._current_position()
            close_enough = (
                current is not None
                and np.linalg.norm(np.asarray(target)[:2] - current) <= 0.35
            )
            if status == GoalStatus.STATUS_SUCCEEDED or close_enough:
                self._publish_treasure(target)
            else:
                self.get_logger().warning('Treasure goal failed; retrying the exact point.')
                self.pending_goal = (np.asarray(target)[:2], True, False)
                self.state = 'WAITING_NAV'
                self.publish_status()
            return
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warning(
                f'Navigation ended with status {status}; starting a local search.'
            )
        self._begin_search()

    def _publish_treasure(self, target: np.ndarray) -> None:
        if self.treasure_published:
            return
        message = PoseStamped()
        message.header.frame_id = 'map'
        message.header.stamp = self.get_clock().now().to_msg()
        message.pose.position.x = float(target[0])
        message.pose.position.y = float(target[1])
        message.pose.orientation.w = 1.0
        self.treasure_pub.publish(message)
        self._store_marker(
            'treasure', 1, Marker.CYLINDER,
            (target[0], target[1], 0.03), (0.60, 0.60, 0.06),
            (1.0, 0.72, 0.02, 1.0),
        )
        self._store_marker(
            'labels', 301, Marker.TEXT_VIEW_FACING,
            (target[0], target[1], 0.55), (0.0, 0.0, 0.30),
            (1.0, 0.72, 0.02, 1.0), 'TREASURE',
        )
        self.treasure_published = True
        self.state = 'DONE'
        self._stop_robot()
        self.publish_status()
        duration = (self.get_clock().now() - self.hunt_started).nanoseconds * 1.0e-9
        self.get_logger().info(
            f'Treasure reached; hunt complete in {duration:.1f} simulated seconds.'
        )

    def _cancel_navigation(self) -> None:
        self.goal_serial += 1
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
            self.goal_handle = None
        self.active_goal = None
        self.goal_started = -1.0

    def _stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def _begin_search(self) -> None:
        self.goal_started = -1.0
        self.active_goal = None
        self.state = 'SEARCHING'
        self.search_started = self.get_clock().now().nanoseconds * 1.0e-9
        self.publish_status()

    def publish_status(self) -> None:
        public_state = self.state
        if public_state in ('WAITING_NAV', 'LOCATING_PILLARS'):
            public_state = 'SEARCHING'
        self.status_pub.publish(String(data=public_state))

    def control_tick(self) -> None:
        if self.state == 'MOVING' and self.goal_started >= 0.0:
            now = self.get_clock().now().nanoseconds * 1.0e-9
            if now - self.goal_started > self.navigation_timeout:
                timed_out_goal = self.active_goal
                self.get_logger().warning(
                    'Navigation timeout reached; cancelling the stuck goal.'
                )
                self._cancel_navigation()
                if timed_out_goal is not None and timed_out_goal[1]:
                    self.pending_goal = timed_out_goal
                    self.state = 'WAITING_NAV'
                    self.publish_status()
                else:
                    self._begin_search()
                return
        if self.state == 'WAITING_NAV' and self.pending_goal is not None:
            if self.navigator.server_is_ready():
                target, treasure, approach = self.pending_goal
                self.pending_goal = None
                self._send_navigation_goal(target, treasure, approach)
            return
        if self.state in ('SEARCHING', 'LOCATING_PILLARS'):
            now = self.get_clock().now().nanoseconds * 1.0e-9
            relocation_delay = (
                18.0
                if self.state == 'LOCATING_PILLARS' or self.search_centre is None
                else 15.0
            )
            if now - self.search_started > relocation_delay:
                if self.state == 'LOCATING_PILLARS' or self.search_centre is None:
                    viewpoint = self._next_exploration_point()
                    if viewpoint is not None:
                        self.pending_goal = (viewpoint, False, False)
                        self.state = 'WAITING_NAV'
                        self._stop_robot()
                        self.publish_status()
                        self.get_logger().info(
                            'No target visible; relocating to map-derived '
                            f'viewpoint ({viewpoint[0]:.2f}, {viewpoint[1]:.2f}).'
                        )
                        return
                elif self.state == 'SEARCHING' and self.search_centre is not None:
                    # Eight viewpoints cover the complete 1.8 m clue
                    # neighbourhood. Each desired point is snapped to a map
                    # cell with sufficient obstacle clearance.
                    for _ in range(8):
                        angle = (
                            self.search_waypoint_index % 8
                        ) * (math.pi / 4.0)
                        self.search_waypoint_index += 1
                        desired = self.search_centre + 1.45 * np.array([
                            math.cos(angle), math.sin(angle)
                        ])
                        viewpoint = self._safe_viewpoint_near(desired)
                        current = self._current_position()
                        if viewpoint is None:
                            continue
                        if current is not None and np.linalg.norm(
                            viewpoint - current
                        ) < 0.45:
                            continue
                        self.pending_goal = (viewpoint, False, False)
                        self.state = 'WAITING_NAV'
                        self._stop_robot()
                        self.publish_status()
                        self.get_logger().info(
                            'Board not visible; moving to local search '
                            f'viewpoint ({viewpoint[0]:.2f}, {viewpoint[1]:.2f}).'
                        )
                        return
                    self.search_started = now
            command = Twist()
            command.angular.z = self.search_speed
            self.cmd_pub.publish(command)
        elif self.state in ('READING', 'DONE'):
            self._stop_robot()


def main():
    rclpy.init()
    node = HuntNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
