from amr_simulation.robot import Robot
from typing import Any


class TurtleBot3Burger(Robot):
    """Class to control the Turtlebot3 Burger robot."""

    # Constants
    SENSOR_RANGE = 8.0  # LiDAR sensor range [m]
    TRACK = 0.16  # Distance between same axle wheels [m]
    WHEEL_RADIUS = 0.033  # Radius of the wheels [m]
    MAX_SPEED = 0.22  # Maximum linear velocity [m/s]

    def __init__(self, sim: Any, dt: float) -> None:
        """Turtlebot3 Burger robot class initializer.

        Args:
            sim: CoppeliaSim simulation handle.
            dt: Sampling period [s].

        """
        Robot.__init__(self, sim=sim, track=self.TRACK, wheel_radius=self.WHEEL_RADIUS)
        self._dt: float = dt
        self._motors: dict[str, int] = self._init_motors()

    def move(self, v: float, w: float) -> None:
        """Solve inverse differential kinematics and send commands to the motors.

        Limits the wheel angular velocities to comply with the maximum linear speed.

        Args:
            v: Linear velocity of the robot center [m/s].
            w: Angular velocity of the robot center [rad/s].

        """
        # TODO: 2.1. Complete the function body with your code (i.e., replace the pass statement).
        # limit linear speed:
        v = min(self.MAX_SPEED, v)
        wr = (v + w * self.TRACK / 2) / self.WHEEL_RADIUS
        wl = (v - w * self.TRACK / 2) / self.WHEEL_RADIUS

        # send commands to the motors
        self._sim.setJointTargetVelocity(self._motors["left"], wl)
        self._sim.setJointTargetVelocity(self._motors["right"], wr)

    def sense(self) -> tuple[list[float], float, float]:
        """Read the LiDAR and the encoders.

        Returns:
            z_scan: Distance from every LiDAR ray to the closest obstacle in 1.5º increments [m].
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read LiDAR
        packed_data: str = self._sim.getStringSignal("lidar")
        z_scan: list[float] = self._sim.unpackFloatTable(packed_data)

        # Return nan if the measurement failed
        z_scan = [z if z >= 0.0 else float("nan") for z in z_scan]

        # Read encoders
        z_v, z_w = self._sense_encoders()

        return z_scan, z_v, z_w

    def _init_motors(self) -> dict[str, int]:
        """Acquire motor handles.

        Returns: {'left': handle, 'right': handle}

        """
        motors: dict[str, int] = {}

        motors["left"] = self._sim.getObject("/leftMotor")
        motors["right"] = self._sim.getObject("/rightMotor")

        return motors

    def _sense_encoders(self) -> tuple[float, float]:
        """Solve forward differential kinematics from encoder readings.

        Returns:
            z_v: Linear velocity of the robot center [m/s].
            z_w: Angular velocity of the robot center [rad/s].

        """
        # Read the angular position increment in the last sampling period [rad]
        encoders: dict[str, float] = {}

        encoders["left"] = self._sim.getFloatSignal("leftEncoder")
        encoders["right"] = self._sim.getFloatSignal("rightEncoder")

        # TODO: 2.2. Compute the derivatives of the angular positions to obtain velocities [rad/s].
        wl = encoders["left"] / self._dt
        wr = encoders["right"] / self._dt

        # TODO: 2.3. Solve forward differential kinematics (i.e., calculate z_v and z_w).
        z_v = (self.WHEEL_RADIUS / 2) * (wl + wr)
        z_w = (self.WHEEL_RADIUS / self.TRACK) * (wr - wl)

        return z_v, z_w
