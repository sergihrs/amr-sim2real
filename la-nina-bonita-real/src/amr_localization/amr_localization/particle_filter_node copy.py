import os
import time

import message_filters
import numpy as np
import rclpy
from amr_msgs.msg import PoseStamped, Stop
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from transforms3d.euler import euler2quat, quat2euler

from amr_localization.particle_filter import ParticleFilter


class ParticleFilterNode(Node):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")
        self.get_logger().info("Initializing particle filter node...")

        # Parameters
        self.declare_parameter("dt", 0.05)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.declare_parameter("enable_plot", False)
        self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value

        self.declare_parameter("particles", 1000)
        particles = self.get_parameter("particles").get_parameter_value().integer_value

        self.declare_parameter("steps_btw_sense_updates", 5)
        steps_btw_sense_updates = (
            self.get_parameter("steps_btw_sense_updates").get_parameter_value().integer_value
        )

        self.declare_parameter("world", "lab03")
        world = self.get_parameter("world").get_parameter_value().string_value

        # Subscribers
        scan_qos_profile = QoSProfile(
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )

        self._subscribers: list[message_filters.Subscriber] = []
        self._subscribers.append(message_filters.Subscriber(self, Odometry, "odom", qos_profile=10))
        self._subscribers.append(
            message_filters.Subscriber(self, LaserScan, "scan", qos_profile=scan_qos_profile)
        )

        ts = message_filters.ApproximateTimeSynchronizer(self._subscribers, queue_size=10, slop=2)
        ts.registerCallback(self._compute_pose_callback)

        # TODO: 3.1. Create the /pose publisher (PoseStamped message).
        self._pose_publisher = self.create_publisher(PoseStamped, "pose", 10)

        # Constants
        SENSOR_RANGE = 3.0  # LiDAR sensor range [m]

        # Attribute and object initializations
        self._localized = False
        self._log_level = self.get_logger().get_effective_level()
        self._steps = 0
        self._steps_btw_sense_updates = steps_btw_sense_updates
        map_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
        )

        self.get_logger().info(f"Initialazing particle filter with {particles} particles.")
        self._particle_filter = ParticleFilter(
            dt,
            map_path,
            SENSOR_RANGE,
            particle_count=particles,
            sigma_v=0.25,
            sigma_w=0.5,
            sigma_z=0.45,
            logger=self.get_logger(),
        )

        self.get_logger().info("Particle filter node initialized!.")

        if self._enable_plot:
            self._particle_filter.show("Initialization", save_figure=True)

        # Stop publisher
        self._stop = self.create_publisher(Stop, "stop", 10)

        # Time between pose
        self._last_pose_time = None
        self._prev_pose = None

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        # Parse measurements
        # z_v: float = odom_msg.twist.twist.linear.x
        # z_w: float = odom_msg.twist.twist.angular.z
        if self._prev_pose and self._last_pose_time:
            pose = odom_msg.pose.pose
            x = pose.position.x
            y = pose.position.y
            _, _, theta = quat2euler(
                (pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z)
            )

            p_x = self._prev_pose.pose.position.x
            p_y = self._prev_pose.pose.position.y
            p_theta = quat2euler(
                (
                    self._prev_pose.pose.orientation.w,
                    self._prev_pose.pose.orientation.x,
                    self._prev_pose.pose.orientation.y,
                    self._prev_pose.pose.orientation.z,
                )
            )[2]

            dt = time.perf_counter() - self._last_pose_time

            z_x = (x - p_x) / dt
            z_y = (y - p_y) / dt
            z_v = np.sqrt(z_x**2 + z_y**2)
            z_w = (theta - p_theta) / dt
            z_w = (z_w + np.pi) % (2 * np.pi) - np.pi

        else:
            z_v = 0
            z_w = 0

        self._last_pose_time = time.perf_counter()
        self._prev_pose = odom_msg.pose

        z_scan: list[float] = scan_msg.ranges

        self.get_logger().info(f"z_v: {z_v}, z_w: {z_w}")

        # Execute particle filter
        self._execute_motion_step(z_v, z_w)
        x_h, y_h, theta_h = self._execute_measurement_step(z_scan)
        self._steps += 1

        # Publish
        self._publish_pose_estimate(x_h, y_h, theta_h)

    def _execute_measurement_step(self, z_scan: list[float]) -> tuple[float, float, float]:
        """Executes and monitors the measurement step (sense) of the particle filter.

        Args:
            z_scan: Distance from every LiDAR ray to the closest obstacle [m].

        Returns:
            Pose estimate (x_h, y_h, theta_h) [m, m, rad]; inf if cannot be computed.
        """
        pose = (float("inf"), float("inf"), float("inf"))
        # TODO Do two moves for every sense when localized
        self.get_logger().info(f"Steps: {self._steps}")

        if self._localized or self._steps % self._steps_btw_sense_updates == 0:
            self._stop.publish(Stop(stop=True))

            start_time = time.perf_counter()
            self._particle_filter.resample(z_scan)
            sense_time = time.perf_counter() - start_time
            self.get_logger().info(f"Sense time: {sense_time}")

            # if self._enable_plot:
            #     self._particle_filter.show("Sense", save_figure=True)

            start_time = time.perf_counter()
            self._localized, pose = self._particle_filter.compute_pose()
            clustering_time = time.perf_counter() - start_time
            clustering_time

            self._stop.publish(Stop(stop=False))

        return pose

    def _execute_motion_step(self, z_v: float, z_w: float):
        """Executes and monitors the motion step (move) of the particle filter.

        Args:
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].
        """
        start_time = time.perf_counter()
        self._particle_filter.move(z_v, z_w)
        move_time = time.perf_counter() - start_time
        move_time

        if self._enable_plot:
            self._particle_filter.show("Move", save_figure=True)

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        pose_msg = PoseStamped()
        pose_msg.localized = self._localized
        if self._localized:
            pose_msg.pose.position.x = x_h
            pose_msg.pose.position.y = y_h
            pose_msg.pose.position.z = 0.0
            quaternion = euler2quat(0, 0, theta_h)

            pose_msg.pose.orientation.w = quaternion[0]
            pose_msg.pose.orientation.x = quaternion[1]
            pose_msg.pose.orientation.y = quaternion[2]
            pose_msg.pose.orientation.z = quaternion[3]

        # publicamos el mensaje
        self._pose_publisher.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    particle_filter_node = ParticleFilterNode()

    try:
        rclpy.spin(particle_filter_node)
    except KeyboardInterrupt:
        pass

    particle_filter_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
