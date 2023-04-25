import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    world = "project"
    start = (-0.8, 0.4, math.pi)
    goal = (1.2, 0.8)
    particles = 2000

    return LaunchDescription(
        [
            Node(
                package="amr_control",
                executable="wall_follower",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[{"enable_localization": True}],
            ),
            Node(
                package="amr_localization",
                executable="particle_filter",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[
                    {"enable_plot": True, "particles": particles, "world": world}
                ],
            ),
            Node(
                package="amr_planning",
                executable="rrt_star",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[{"goal": goal, "world": world}],
            ),
            Node(
                package="amr_control",
                executable="pure_pursuit",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
            ),
            # Node(
            #     package="amr_simulation",
            #     executable="coppeliasim",
            #     output="screen",
            #     arguments=["--ros-args", "--log-level", "INFO"],
            #     parameters=[
            #         {"enable_localization": True, "start": start, "goal": goal}
            #     ],
            # ),  # Must be launched last
        ]
    )
