from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

from amr_msgs.msg import PoseStamped
from rosidl_runtime_py import message_to_yaml

import math
from transforms3d.euler import euler2quat


def generate_launch_description():
    start = (-4.0, -4.0, math.pi / 2)

    msg = PoseStamped()
    msg.localized = True
    quat_w, quat_x, quat_y, quat_z = euler2quat(0.0, 0.0, start[2])
    msg.pose.position.x = start[0]
    msg.pose.position.y = start[1]
    msg.pose.orientation.w = quat_w
    msg.pose.orientation.x = quat_x
    msg.pose.orientation.y = quat_y
    msg.pose.orientation.z = quat_z

    return LaunchDescription(
        [
            ExecuteProcess(
                name="publish_pose",
                cmd=[
                    "ros2",
                    "topic",
                    "pub",
                    "--once",
                    "/pose",
                    "amr_msgs/msg/PoseStamped",
                    message_to_yaml(msg),
                ],
                output="screen",
            ),
        ]
    )
