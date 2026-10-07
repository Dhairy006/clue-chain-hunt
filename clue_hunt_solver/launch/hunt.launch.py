"""
Starts YOUR leader and follower nodes. This is the single command used during evaluation
(after the simulation + Nav2 are up):

  ros2 launch clue_hunt_solver hunt.launch.py

Add parameters / extra nodes here as you need them.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('clue_hunt_solver'),
        'config',
        'solver_params.yaml',
    )
    return LaunchDescription([
        Node(package='clue_hunt_solver', executable='hunt_node', output='screen',
             parameters=[params, {'use_sim_time': True}]),
        Node(package='clue_hunt_solver', executable='follower_node', output='screen',
             parameters=[params, {'use_sim_time': True}]),
    ])
