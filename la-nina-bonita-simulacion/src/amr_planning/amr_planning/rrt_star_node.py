import os

import rclpy
from amr_msgs.msg import PoseStamped as AmrPoseStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node

from amr_planning.rrt_star import AStar


class RRTStarNode(Node):
    def __init__(self):
        """RRT* node initializer."""
        super().__init__("rrt_star")

        # Parameters
        self.declare_parameter("goal", (0.0, 0.0))
        self._goal = tuple(
            self.get_parameter("goal").get_parameter_value().double_array_value.tolist()
        )

        self.declare_parameter("world", "lab04")
        world = self.get_parameter("world").get_parameter_value().string_value

        # Subscribers
        self._subscriber_pose = self.create_subscription(
            AmrPoseStamped, "pose", self._path_callback, 10
        )

        # TODO: 4.6. Create the /path publisher (Path message).
        self._publisher_path = self.create_publisher(Path, "path", qos_profile=10)

        # Constants
        SENSOR_RANGE = 8.0  # LiDAR sensor range [m]

        # Attribute and object initializations
        map_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
        )
        self._localized = False
        # self._planning = RRTStar(
        #     map_path, SENSOR_RANGE, safety_distance=0.15, logger=self.get_logger()
        # )
        self._planning = AStar(
            map_path, SENSOR_RANGE, safety_distance=0.17, logger=self.get_logger()
        )

    def _path_callback(self, pose_msg: AmrPoseStamped):
        """Subscriber callback. Executes RRT* and publishes the smoothed path to the goal.

        Args:
            pose_msg: Message containing the robot pose estimate.

        """
        if pose_msg.localized and not self._localized:
            # Execute RRT*
            start = (pose_msg.pose.position.x, pose_msg.pose.position.y)
            path = self._planning.compute_path(start, self._goal)
            smoothed_path = self._planning.smooth_path(
                path, data_weight=0.3, smooth_weight=0.05, tolerance=1e-3
            )
            smoothed_path = path

            self._planning.show(path, smoothed_path, save_figure=True)
            self._publish_path(smoothed_path)

        self._localized = pose_msg.localized

    def _publish_path(self, path: list[tuple[float, float]]) -> None:
        """Publishes the robot's path to the goal in a nav_msgs.msg.Path message.

        Args:
            path: Smoothed path (initial location first) in (x, y) format.

        """
        # TODO: 4.7. Complete the function body with your code (i.e., replace the pass statement).
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for x, y in path:
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            path_msg.poses.append(pose)
        self._publisher_path.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)

    rrt_star_node = RRTStarNode()

    try:
        rclpy.spin(rrt_star_node)
    except KeyboardInterrupt:
        pass

    rrt_star_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
