import rclpy
from rclpy.node import Node

from amr_msgs.msg import Keypress

import sshkeyboard


class KeyboardNode(Node):
    def __init__(self):
        super().__init__("keyboard_node")
        self._publisher = self.create_publisher(
            msg_type=Keypress, topic="keyboard", qos_profile=10
        )
        self.get_logger().info("Initializing keyboard node...")
        sshkeyboard.listen_keyboard(on_press=self.publish_keyboard)

    def publish_keyboard(self, key):
        msg = Keypress()
        msg.key = key
        self._publisher.publish(msg)
        self.get_logger().info(f"Published key: {key}")


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardNode()
    rclpy.spin(node)
    rclpy.try_shutdown()
