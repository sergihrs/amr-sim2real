import rclpy
from rclpy.node import Node

from amr_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path

import math
from transforms3d.euler import quat2euler

from amr_control.pure_pursuit import PurePursuit


class PurePursuitNode(Node):
    def __init__(self):
        """Pure pursuit node initializer."""
        super().__init__("pure_pursuit")
        self.get_logger().info("Initializing pure pursuit node...")

        # Parameters
        self.declare_parameter("dt", 0.05)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.declare_parameter("lookahead_distance", 0.15)
        lookahead_distance = (
            self.get_parameter("lookahead_distance").get_parameter_value().double_value
        )

        # Subscribers
        self._subscriber_pose = self.create_subscription(
            PoseStamped, "pose", self._compute_commands_callback, 10
        )
        self._subscriber_path = self.create_subscription(
            Path, "path", self._path_callback, 10
        )

        # Publishers
        self._publisher = self.create_publisher(
            msg_type=Twist, topic="cmd_vel", qos_profile=10
        )

        # Attribute and object initializations
        self._pure_pursuit = PurePursuit(dt, lookahead_distance, self.get_logger())

    def _compute_commands_callback(self, pose_msg: PoseStamped):
        """Subscriber callback. Executes a pure pursuit controller and publishes v and w commands.

        Starts to operate once the robot is localized.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """

        self._logger.info(f"POSE RECEIVED: {pose_msg.localized}")

        if pose_msg.localized:
            # Parse pose
            x = pose_msg.pose.position.x
            y = pose_msg.pose.position.y
            quat_w = pose_msg.pose.orientation.w
            quat_x = pose_msg.pose.orientation.x
            quat_y = pose_msg.pose.orientation.y
            quat_z = pose_msg.pose.orientation.z
            # self.get_logger().info(f"RECIEVED QUATERNION: {quat_w:.3f}, {quat_x:.3f}, {quat_y:.3f}, {quat_z:.3f}")
            _, _, theta = quat2euler((quat_w, quat_x, quat_y, quat_z))
            # self.get_logger().info(f"RECIEVED THETA: {theta:.3f}")
            theta %= 2 * math.pi

            # Execute pure pursuit
            v, w = self._pure_pursuit.compute_commands(x, y, theta)
            self.get_logger().info(f"CONTROL WITH PURE PURSUIT")
            self.get_logger().info(f"Commands: v = {v:.3f} m/s, w = {w:+.3f} rad/s")

            # Publish
            self._publish_velocity_commands(v, -w)

    def _path_callback(self, path_msg: Path):
        """Subscriber callback. Saves the path the pure pursuit controller has to follow.

        Args:
            path_msg: Message containing the (smoothed) path.

        """
        # TODO: 4.8. Complete the function body with your code (i.e., replace the pass statement).
        self.get_logger().info("PATH CREATED!!!!")
        pose: PoseStamped
        self._pure_pursuit.path = [
            (pose.pose.position.x, pose.pose.position.y) for pose in path_msg.poses
        ]

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    pure_pursuit_node = PurePursuitNode()

    try:
        rclpy.spin(pure_pursuit_node)
    except KeyboardInterrupt:
        pass

    pure_pursuit_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
