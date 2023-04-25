from time import sleep
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

import message_filters
from amr_msgs.msg import PoseStamped, Stop
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from amr_control.wall_follower import WallFollower


class WallFollowerNode(Node):
    def __init__(self):
        """Wall follower node initializer."""

        # sleep(15)

        super().__init__("wall_follower")
        self.get_logger().info("Initializing wall follower node...")

        # Parameters
        self.declare_parameter("dt", 0.05)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.declare_parameter("enable_localization", False)
        _ = self.get_parameter("enable_localization").get_parameter_value().bool_value

        # TODO: 2.7. Subscribe to /odom and /scan and sync them with _compute_commands_callback.
        # sync nodes
        scan_qos_profile = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
        )
        self._subscriber: list[message_filters.Subscriber] = []
        self._subscriber.append(
            message_filters.Subscriber(self, Odometry, "odom")
        )
        self._subscriber.append(
            message_filters.Subscriber(self, LaserScan, "scan", qos_profile=scan_qos_profile)
        )

        ts = message_filters.ApproximateTimeSynchronizer(
            self._subscriber, queue_size=10, slop=2
        )
        ts.registerCallback(self._compute_commands_callback)

        # Pose
        self._pose_sub = self.create_subscription(
            PoseStamped, "pose", self._update_pose_msg, 10
        )
        self._pose_msg = PoseStamped()

        # STOP
        self._stop_sub = self.create_subscription(
            Stop, "stop", self._stop_callback, 10
        )
        self.stopped = False

        # TODO: 2.10. Create the /cmd_vel velocity commands publisher (Twist message).
        self._cmd_vel_pub = self.create_publisher(
            msg_type=Twist, topic="cmd_vel", qos_profile=10
        )

        # Attribute and object initializations
        self._wall_follower = WallFollower(dt, self.get_logger())

    def _update_pose_msg(self, pose_msg: PoseStamped):
        """Updates the robot pose with the latest PoseStamped message.

        Args:
            pose_msg: Message containing the estimated robot pose.

        """
        self._pose_msg = pose_msg

    def _stop_callback(self, stop_msg: Stop):
        """Subscriber callback. Stops the robot.

        Args:
            stop_msg: Message containing the stop command.

        """
        if stop_msg.stop:
            self._publish_velocity_commands(0.0, 0.0)
            self.stopped = True
        else:
            self.stopped = False

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
        if not self._pose_msg.localized and not self.stopped:
            # TODO: 2.8. Parse the odometry from the Odometry message (i.e., read z_v and z_w).
            z_v: float = odom_msg.twist.twist.linear.x
            z_w: float = odom_msg.twist.twist.angular.z

            # TODO: 2.9. Parse LiDAR measurements from the LaserScan message (i.e., read z_scan).
            z_scan: list[float] = scan_msg.ranges

            # Execute wall follower
            v, w = self._wall_follower.compute_commands(z_scan, z_v, z_w)

            # self.get_logger().info(f"v: {v}, w: {w}")

            # Publish
            self._publish_velocity_commands(v, -w)

    def _publish_velocity_commands(self, v: float, w: float) -> None:
        """Publishes velocity commands in a geometry_msgs.msg.TwistStamped message.

        Args:
            v: Linear velocity command [m/s].
            w: Angular velocity command [rad/s].

        """
        # TODO: 2.11. Complete the function body with your code (i.e., replace the pass statement).
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w

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
