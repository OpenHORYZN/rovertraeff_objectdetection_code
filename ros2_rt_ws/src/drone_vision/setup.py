from setuptools import find_packages, setup
import os

package_name='drone_vision'
config_file_source = os.path.join('../../..', 'config.py')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/drone_vision/launch', ['launch/drone_vision.launch.py']),
        (os.path.join('share', package_name), [config_file_source])
    ],
    entry_points={
        'console_scripts': [
            'camera_node = drone_vision.camera_node:main',
            'yolo_node = drone_vision.yolo_node:main',
            'aruco_node = drone_vision.aruco_node:main',
            'pose_estimator_node = drone_vision.pose_estimator_node:main',
        ],
    },
)
