from setuptools import find_packages, setup
import os

package_name='drone_vision'
config_file_source = os.path.join('../../..', 'config.py')
weights_file_source = os.path.join('../../..', 'data/runs/FT1TVT1_yolo11_train/weights/best.pt')
marker_positions_source = os.path.join('../../..', 'positioning/aruco_pos.yaml')

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/drone_vision.launch.py']),
        ('share/' + package_name + '/', [config_file_source]),
        ('share/' + package_name + '/data/runs/FT1TVT1_yolo11_train/weights/', [weights_file_source]),
        ('share/' + package_name + '/positioning/', [marker_positions_source])
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
