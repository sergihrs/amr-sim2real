import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # start = (1.15, -1.20, 0.5 * math.pi)  # Outer corridor
    start = (0.55, -0.60, 1.5 * math.pi)  # Inner corridor

    return LaunchDescription(
        [
            # Node(
            #     package="amr_control",
            #     executable="wall_follower",
            #     output="screen",
            #     arguments=["--ros-args", "--log-level", "INFO"],
            # ),
            Node(
                package="amr_control",
                executable="keyboard_node",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
            ),
            Node(
                package="amr_control",
                executable="teleoperation_node",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
            ),
            # Node(
            #     package="amr_simulation",
            #     executable="coppeliasim",
            #     output="screen",
            #     arguments=["--ros-args", "--log-level", "INFO"],
            #     parameters=[{"start": start}],
            # ),  # Must be launched last
        ]
    )
