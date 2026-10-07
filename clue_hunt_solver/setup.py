from glob import glob
import os

from setuptools import setup

package_name = 'clue_hunt_solver'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name), ['README.md']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dhairy006',
    maintainer_email='dhairy006@users.noreply.github.com',
    description='Autonomous leader and camera-only follower for Clue Chain Hunt',
    license='MIT',
    entry_points={
        'console_scripts': [
            'hunt_node = clue_hunt_solver.hunt_node:main',
            'follower_node = clue_hunt_solver.follower_node:main',
        ],
    },
)
