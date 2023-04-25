import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

import message_filters
from amr_msgs.msg import PoseStamped
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from amr_control.wall_follower import WallFollower


class WallFollowerNode(Node):
    def __init__(self):
        """Wall follower node initializer."""
        super().__init__("wall_follower")

        # Parameters
        self.declare_parameter("dt", 0.05)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.declare_parameter("enable_localization", False)
        _ = self.get_parameter("enable_localization").get_parameter_value().bool_value

        # TODO: 2.7. Subscribe to /odom and /scan and sync them with _compute_commands_callback.
        # sync nodes
        scan_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._subscriber: list[message_filters.Subscriber] = []
        self._subscriber.append(
            message_filters.Subscriber(self, Odometry, "odom", qos_profile=10)
        )
        self._subscriber.append(
            message_filters.Subscriber(self, LaserScan, "scan", qos_profile=10)
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            self._subscriber, queue_size=10, slop=2
        )
        ts.registerCallback(self._compute_commands_callback)

        self._pose_sub = self.create_subscription(
            PoseStamped, "pose", self._update_pose_msg, 10
        )
        self._pose_msg = PoseStamped()

        # TODO: 2.10. Create the /cmd_vel velocity commands publisher (TwistStamped message).
        self._cmd_vel_pub = self.create_publisher(
            msg_type=TwistStamped, topic="cmd_vel", qos_profile=10
        )

        # Attribute and object initializations
        self._wall_follower = WallFollower(dt, self.get_logger())

    def _update_pose_msg(self, pose_msg: PoseStamped):
        """Updates the robot pose with the latest PoseStamped message.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """
        # If I was localized and now I'm not, reset the wall follower
        if self._pose_msg.localized and not pose_msg.localized:
            self._wall_follower = WallFollower(
                self._wall_follower._dt, self.get_logger()
            )
        self._pose_msg = pose_msg

    def _compute_commands_callback(
        self,
        odom_msg: Odometry,
        scan_msg: LaserScan,
    ):
        """Subscriber callback. Executes a wall-following controller and publishes v and w commands.

        Ceases to operate once the robot is localized.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR readings.
            pose_msg: Message containing the estimated robot pose.

        """
        if not self._pose_msg.localized:
            # TODO: 2.8. Parse the odometry from the Odometry message (i.e., read z_v and z_w).
            z_v: float = odom_msg.twist.twist.linear.x
            z_w: float = odom_msg.twist.twist.angular.z

            # TODO: 2.9. Parse LiDAR measurements from the LaserScan message (i.e., read z_scan).
            z_scan: list[float] = scan_msg.ranges

            # Execute wall follower
            v, w = self._wall_follower.compute_commands(z_scan, z_v, z_w)

            # Publish
            self._publish_velocity_commands(v, w)

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        # TODO: 2.11. Complete the function body with your code (i.e., replace the pass statement).
        msg = TwistStamped()
        msg.twist.linear.x = v
        msg.twist.angular.z = w

        # publish
        self._cmd_vel_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    wall_follower_node = WallFollowerNode()

    try:
        rclpy.spin(wall_follower_node)
    except KeyboardInterrupt:
        pass

    wall_follower_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
