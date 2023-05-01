import os
import time

import message_filters
import rclpy
from amr_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from transforms3d.euler import euler2quat

from amr_localization.particle_filter import ParticleFilter


class ParticleFilterNode(Node):
    def __init__(self):
        """Particle filter node initializer."""
        super().__init__("particle_filter")

        # Parameters
        self.declare_parameter("dt", 0.05)
        dt = self.get_parameter("dt").get_parameter_value().double_value

        self.declare_parameter("enable_plot", False)
        self._enable_plot = self.get_parameter("enable_plot").get_parameter_value().bool_value

        self.declare_parameter("n_particles", 1000)
        n_particles = self.get_parameter("n_particles").get_parameter_value().integer_value

        self.declare_parameter("steps_btw_sense_updates", 10)
        steps_btw_sense_updates = (
            self.get_parameter("steps_btw_sense_updates").get_parameter_value().integer_value
        )
        self.declare_parameter("steps_btw_sense_updates_localized", 5)
        steps_btw_sense_updates_localized = (
            self.get_parameter("steps_btw_sense_updates_localized")
            .get_parameter_value()
            .integer_value
        )

        self.declare_parameter("world", "lab03")
        world = self.get_parameter("world").get_parameter_value().string_value

        # TurtleBot3 Burger width is 178 mm
        self.declare_parameter("robot_center_clearance", 0.089)
        robot_center_clearance = (
            self.get_parameter("robot_center_clearance").get_parameter_value().double_value
        )

        # Subscribers
        self._subscribers: list[message_filters.Subscriber] = []
        self._subscribers.append(message_filters.Subscriber(self, Odometry, "odom", qos_profile=10))
        self._subscribers.append(
            message_filters.Subscriber(self, LaserScan, "scan", qos_profile=10)
        )

        ts = message_filters.ApproximateTimeSynchronizer(self._subscribers, queue_size=10, slop=2)
        ts.registerCallback(self._compute_pose_callback)

        self._pose_publisher = self.create_publisher(PoseStamped, "pose", 10)

        # Constants
        SENSOR_RANGE = 8.0  # LiDAR sensor range [m]

        # Attribute and object initializations
        self._log_level = self.get_logger().get_effective_level()
        self._steps = 0
        self._steps_btw_sense_updates = max(1, steps_btw_sense_updates)
        self._steps_btw_sense_updates_localized = max(1, steps_btw_sense_updates_localized)
        self._last_pose = (float("inf"), float("inf"), float("inf"))
        map_path = os.path.realpath(
            os.path.join(os.path.dirname(__file__), "..", "maps", world + ".json")
        )

        self._particle_filter = ParticleFilter(
            dt,
            map_path,
            SENSOR_RANGE,
            particle_count=n_particles,
            robot_center_clearance=robot_center_clearance,
            logger=self.get_logger(),
        )

        if self._enable_plot:
            self._particle_filter.show("Initialization", save_figure=True)

    def _compute_pose_callback(self, odom_msg: Odometry, scan_msg: LaserScan):
        """Subscriber callback. Executes a particle filter and publishes (x, y, theta) estimates.

        Args:
            odom_msg: Message containing odometry measurements.
            scan_msg: Message containing LiDAR sensor readings.

        """
        # Parse measurements
        z_v: float = odom_msg.twist.twist.linear.x
        z_w: float = odom_msg.twist.twist.angular.z
        z_scan: list[float] = scan_msg.ranges

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
        sense_interval = (
            self._steps_btw_sense_updates_localized
            if self._particle_filter.localized
            else self._steps_btw_sense_updates
        )
        should_sense = self._steps % sense_interval == 0

        if not should_sense:
            return self._last_pose

        start_time = time.perf_counter()
        self._particle_filter.resample(z_scan)
        sense_time = time.perf_counter() - start_time
        self._logger.info(f"Executed sense step in {sense_time:.2f} seconds")

        if self._enable_plot:
            # Save images only on sense updates to reduce I/O overhead.
            start_time = time.perf_counter()
            self._particle_filter.show("Sense", save_figure=True)
            plot_time = time.perf_counter() - start_time
            self._logger.info(f"Saved particle filter plot in {plot_time:.2f} seconds")

        start_time = time.perf_counter()
        pose = self._particle_filter.compute_pose()
        clustering_time = time.perf_counter() - start_time
        self._logger.info(f"Computed pose estimate in {clustering_time:.2f} seconds")

        self._last_pose = pose
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

        # Move-step image saving intentionally disabled.

    def _publish_pose_estimate(self, x_h: float, y_h: float, theta_h: float) -> None:
        """Publishes the robot's pose estimate in a custom amr_msgs.msg.PoseStamped message.

        Args:
            x_h: x coordinate estimate [m].
            y_h: y coordinate estimate [m].
            theta_h: Heading estimate [rad].

        """
        pose_msg = PoseStamped()
        pose_msg.localized = self._particle_filter.localized
        if self._particle_filter.localized:
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
