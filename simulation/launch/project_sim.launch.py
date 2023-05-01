import math

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    world = "project_sim"

    # Corner, initial
    start = (-0.8, 0.4, math.pi)
    goal = (1.2, 1.2)

    # bottom left
    # start = (-1.2, -1.2, 0)
    # goal = (-1.2, 1.2)

    # bottom right to middle up
    # start = (1.2, -1.2, math.pi / 2)
    # goal = (-0.4, 1.2)

    # top right, facing wall
    # start = (1.2, 1.2, 3 * math.pi / 2)
    # goal = (1.2, -1.2)

    # corner initial facing wall
    # start = (-0.8, 0.4, 3 * math.pi / 2)
    # goal = (1.2, 0.8)

    n_particles = 1500
    enable_plots = True

    return LaunchDescription(
        [
            Node(
                package="amr_control",
                executable="wall_follower",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[{"enable_localization": False}],
            ),
            Node(
                package="amr_localization",
                executable="particle_filter",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[
                    {
                        "n_particles": n_particles,
                        "world": world,
                        "enable_plot": enable_plots,
                        "robot_center_clearance": 0.089,
                    }
                ],
            ),
            Node(
                package="amr_planning",
                executable="rrt_star",
                output="screen",
                arguments=["--ros-args", "--log-level", "INFO"],
                parameters=[{"goal": goal, "world": world, "enable_plot": enable_plots}],
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
