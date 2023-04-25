import math
from time import time
import numpy as np

LINEAR_SPEED = 0.13
ANGULAR_SPEED = 1.0

DESIRED_DISTANCE = 0.22 # 0.2
DESIRED_THRESHOLD = 0.04

NINA_WIDTH = 0.2
NINA_HEIGHT = 0.15

LOGGER = None

def raycast(
    z_scan: list[float], target_angle: float, width: int
) -> tuple[float, float]:
    """
    Raycast function to detect obstacles in a given direction.
    Parameters
    ----------
    z_scan : list[float]
        Distance from every LiDAR ray to the closest obstacle [m].
    target_angle : int
        Angle to the target direction [deg].
    width : int
        Number of LiDAR rays to consider in the raycast.

    Returns
    -------
    float: Distance to the closest obstacle [m].
    float: Angle to the closest obstacle [rad].
    """
    # LOGGER.info("z_scan: " + str(z_scan))

    # Convert to numpy array
    z_scan = np.array(z_scan)

    # Replace nans with 8
    z_scan[np.isnan(z_scan)] = 8

    target_index = int(target_angle * len(z_scan) / 360)

    # Get the indexes of the rays to consider
    indexes = np.arange(target_index - width // 2, target_index + width // 2 + 1) % len(
        z_scan
    )

    # Get the distances of the rays to consider
    distances = z_scan[indexes]

    # Get the index of the closest obstacle and minimum distance
    argmin_index = np.argmin(distances)
    min_distance = distances[argmin_index]

    # Get the angle of the closest obstacle
    argmin_angle = indexes[argmin_index] * 2 * math.pi / len(z_scan)

    return min_distance, argmin_angle

def get_rays(
    z_scan: list[float], target_angle: float, width: int
) -> tuple[list[float], list[int]]:
    """
    Function to get the rays in a given direction.
 
    Parameters
    ----------
    z_scan : list[float]
        Distance from every LiDAR ray to the closest obstacle [m].
    target_angle : int
        Angle to the target direction [deg].
    width : int
        Number of LiDAR rays to return.
 
    Returns
    -------
    list[float]: Distances to the closest obstacle [m].
    list[int]: Indexes of the rays.
    """
 
    z_scan = np.array(z_scan)
 
    z_scan[np.isnan(z_scan)] = 8
 
    target_index = int(target_angle * len(z_scan) / 360)
 
    # Get the indexes of the rays to consider
    indexes = np.arange(target_index - width // 2, target_index + width // 2 + 1) % len(
        z_scan
    )
 
    # Get the distances of the rays to consider
    distances = z_scan[indexes]
 
    return distances, indexes
 

class RobotState:
    def __init__(self) -> None:
        self._delta_time: float = 0.0
        self._last_time = time()
        LOGGER.info(f"Entering {self.__class__.__name__}")

    @staticmethod
    def execute_decorator(execute):
        def wrapper(self, z_scan: list[float], z_v: float, z_w: float):
            self._delta_time = time() - self._last_time
            self._last_time = time()
            # self._delta_time = 0.05
            # LOGGER.info(f"delta_time: {self._delta_time:.4f}")
            return execute(self, z_scan, z_v, z_w)

        return wrapper


class InitialState(RobotState):
    def __init__(self) -> None:
        super().__init__()
 
    def exit_state(self, z_scan: list[float]) -> None:
        if all(ray < 0.4 for ray in get_rays(z_scan, 270, 11)[0]):
            return FollowWall()
 
    @RobotState.execute_decorator
    def execute(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        return 0.0, ANGULAR_SPEED

class FollowWall(RobotState):
    def __init__(self):
        super().__init__()
        self._previous_error = 0

    def exit_state(self, z_scan: list[float]) -> None:
        if raycast(z_scan, 0, 3)[0] < DESIRED_DISTANCE:
            return OuterTurn()

        if raycast(z_scan, 270, 3)[0] > 0.4:
            return InnerTurn()

    @RobotState.execute_decorator
    def execute(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        """
        Proportional controller to follow the wall at a constant distance
        and parallel to it.
        """

        min_distance, argmin_angle = raycast(z_scan, 270, 69)

        distance_error = DESIRED_DISTANCE - min_distance
        angle_error = argmin_angle - 3 * math.pi / 2

        kp1, kd1, kp2 = 10, 1, 1
        w_pd = (
            kp1 * distance_error
            + kd1 * (distance_error - self._previous_error) / self._delta_time
        )
        w = w_pd + kp2 * angle_error

        # LOGGER.info(f"min_distance: {min_distance:.4f}, w: {w:.4f}")
        # LOGGER.info(f"previous_error: {self._previous_error:.4f}, delta_time: {self._delta_time:.4f}")

        self._previous_error = distance_error

        # Saturate w
        w = min(max(w, -1.0), 1.0)

        return LINEAR_SPEED, w


class OuterTurn(RobotState):
    def __init__(self, end_angle: float = math.pi / 2) -> None:
        super().__init__()
        self.END_ANGLE = end_angle
        self._current_angle = 0

    def exit_state(self, z_scan: list[float]) -> None:
        if self._current_angle >= self.END_ANGLE:
            return FollowWall()

    @RobotState.execute_decorator
    def execute(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        """
        Turn the robot until the angle is reached approximately.
        """

        self._current_angle += ANGULAR_SPEED * self._delta_time

        return 0.0, ANGULAR_SPEED


class InnerTurn(RobotState):
    def __init__(self, end_angle: float = math.pi / 2) -> None:
        super().__init__()

        self.END_ANGLE = end_angle
        self._current_distance = 0
        self._current_angle = 0

        self._step = 0

    def exit_state(self, z_scan: list[float]) -> None:
        if (
            self._step == 2
            and raycast(z_scan, 270, 3)[0] < DESIRED_DISTANCE + DESIRED_THRESHOLD
        ):
            return FollowWall()
        elif raycast(z_scan, 0, 3)[0] < DESIRED_DISTANCE:
            return OuterTurn()

    @RobotState.execute_decorator
    def execute(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        """
        1. Move forward to surpass the corner. (0.35m)
        2. Turn right until the angle is reached approximately.
        3. Move until the right wall is detected.
        """
        if self._step == 0:
            self._current_distance += LINEAR_SPEED * self._delta_time

            if self._current_distance >= DESIRED_DISTANCE - NINA_HEIGHT / 2:
                self._step += 1
                self._current_distance = 0

            return LINEAR_SPEED, 0.0

        elif self._step == 1:
            self._current_angle += ANGULAR_SPEED * self._delta_time

            if self._current_angle >= self.END_ANGLE:
                self._step += 1
                self._current_angle = 0

            return 0.0, -ANGULAR_SPEED

        elif self._step == 2:
            return LINEAR_SPEED, 0.0


class WallFollower:
    """Class to safely explore an environment (without crashing) when the pose is unknown."""

    def __init__(self, dt: float, logger=None) -> None:
        """Wall following class initializer.

        Args:
            dt: Sampling period [s].
        """
        self._dt: float = dt

        self.logger = logger

        global LOGGER
        LOGGER = logger

        self.state = InitialState()

        # LOGGER.info(f"dt: {self._dt}")

    def compute_commands(
        self, z_scan: list[float], z_v: float, z_w: float
    ) -> tuple[float, float]:
        """Wall following exploration algorithm.

        Args:
            z_scan: Distance from every LiDAR ray to the closest obstacle [m].
            z_v: Odometric estimate of the linear velocity of the robot center [m/s].
            z_w: Odometric estimate of the angular velocity of the robot center [rad/s].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 1.14. Complete the function body with your code (i.e., compute v and w).
        # self.logger.info(f"ODOMETRY::: v: {z_v}, w: {z_w}")

        if (new_state := self.state.exit_state(z_scan)) is not None:
            self.state = new_state

        v, w = self.state.execute(z_scan, z_v, z_w)

        return v, w
