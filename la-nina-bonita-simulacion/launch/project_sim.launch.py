import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    world = "project_sim"

    # Corner, initial
    start = (-0.8, 0.4, math.pi)
    goal = (1.2, 0.8)

    # bottom left
    # start = (-1.2, -1.2, 0)
    # goal = (-1.2, 1.2)

    # top right, facing wall
    # start = (1.2, 1.2, 3 * math.pi / 2)
    # goal = (1.2, -1.2)

    # corner initial facing wall
    # start = (-0.8, 0.4, 3 * math.pi / 2)
    # goal = (1.2, 0.8)

    particles = 2500

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
                parameters=[{"particles": particles, "world": world, "enable_plot": True}],
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
            Node(
                package="amr_simulation",
                executable="coppeliasim",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[{"enable_localization": True, "start": start, "goal": goal}],
            ),  # Must be launched last
        ]
    )
