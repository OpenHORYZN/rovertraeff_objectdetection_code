from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # Not needed with D455 realsense camera,
        # which already has a camera topic
        # Node(
        #     package='drone_vision',
        #     executable='camera_node',
        #     name='camera_node',
        #     output='screen',
        # ),
        Node(
            package='drone_vision',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
        ),
        Node(
            package='drone_vision',
            executable='aruco_node',
            name='aruco_node',
            output='screen',
        ),
        Node(
            package='drone_vision',
            executable='pose_estimator_node',
            name='pose_estimator_node',
            output='screen',
        ),
    ])