import math
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)

from amr_msgs.msg import Keypress

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class TeleoperationNode(Node):
    def __init__(self):
        super().__init__("teleoperation_node")
        self._subscriber_keyboard = self.create_subscription(
            msg_type=Keypress,
            topic="keyboard",
            callback=self.callback_keyboard,
            qos_profile=10,
        )
        self._publisher = self.create_publisher(
            msg_type=Twist, topic="cmd_vel", qos_profile=10
        )
        self.get_logger().info("Teleoperation node has been started.")

        # Linear velocity
        self._linear_velocity = 0.0
        # Angular velocity
        self._angular_velocity = 0.0

    def callback_keyboard(self, msg):
        self.get_logger().info(f"Received key: {msg.key}")
        if msg.key == "w":
            self._linear_velocity += 0.1
        elif msg.key == "s":
            self._linear_velocity -= 0.1
        elif msg.key == "a":
            self._angular_velocity -= 0.3
        elif msg.key == "d":
            self._angular_velocity += 0.3
        elif msg.key == "space":
            self._linear_velocity = 0.0
            self._angular_velocity = 0.0

        self.publish_twist()

    def publish_twist(self):
        msg = Twist()
        msg.linear.x = self._linear_velocity
        msg.angular.z = self._angular_velocity
        self._publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeleoperationNode()
    rclpy.spin(node)
    rclpy.try_shutdown()
