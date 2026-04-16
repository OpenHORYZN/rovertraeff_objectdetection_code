from setuptools import find_packages, setup

setup(
    name='drone_vision',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/drone_vision']),
        ('share/drone_vision', ['package.xml']),
        ('share/drone_vision/launch', ['launch/drone_vision.launch.py']),
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
