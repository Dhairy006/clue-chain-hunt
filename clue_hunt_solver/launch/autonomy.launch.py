"""One-command simulation, Nav2, leader, and follower launch."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    navigation_share = get_package_share_directory('clue_hunt_navigation')
    solver_share = get_package_share_directory('clue_hunt_solver')

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(navigation_share, 'launch', 'navigation.launch.py')
        ),
        launch_arguments={
            'map': LaunchConfiguration('map'),
            'gui': LaunchConfiguration('gui'),
            'rviz': LaunchConfiguration('rviz'),
            'follower': 'true',
            'sim': 'true',
        }.items(),
    )
    solver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(solver_share, 'launch', 'hunt.launch.py')
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(solver_share, 'maps', 'arena.yaml'),
            description='Absolute path to the saved arena map YAML',
        ),
        DeclareLaunchArgument('gui', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        navigation,
        TimerAction(period=12.0, actions=[solver]),
    ])
