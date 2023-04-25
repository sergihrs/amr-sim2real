import numpy as np


class PurePursuit:
    """Class to follow a path using a simple pure pursuit controller."""

    def __init__(self, dt: float, lookahead_distance: float = 0.5, logger=None):
        """Pure pursuit class initializer.

        Args:
            dt: Sampling period [s].
            lookahead_distance: Distance to the next target point [m].

        """
        self._dt: float = dt
        self._lookahead_distance: float = lookahead_distance
        self._path: list[tuple[float, float]] = []

        self._logger = logger

    def compute_commands(self, x: float, y: float, theta: float) -> tuple[float, float]:
        """Pure pursuit controller implementation.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].
            theta: Estimated robot heading [rad].

        Returns:
            v: Linear velocity [m/s].
            w: Angular velocity [rad/s].

        """
        # TODO: 4.11. Complete the function body with your code (i.e., compute v and w).
        if not self._path:
            return 0.0, 0.0

        v = 0.09  # MAX SPEED

        closest_xy, closest_idx = self._find_closest_point(x, y)
        target_xy = self._find_target_point([x, y], closest_idx)

        self._logger.info(f"Closest point: {closest_xy}")
        self._logger.info(f"Closest point: {target_xy}")

        l = np.linalg.norm(np.array(target_xy) - np.array([x, y]))
        beta = np.arctan2(target_xy[1] - y, target_xy[0] - x)
        beta %= 2 * np.pi
        alpha = beta - theta

        # Steer when angle is too large
        if abs(alpha) > np.pi / 5:
            self._logger.info(f"Steering too much:{alpha}")
            return 0.0, np.sign(np.sin(alpha)) * np.pi / 4

        # If reached goal, stop and celebrate 🎉
        if np.isclose(l, 0.0, atol=1e-4):
            self._logger.info("GOAL REACHED!")
            v, w = 0.0, 4 * np.pi
        else:
            w = 2 * np.sin(alpha) * v / l

        return v, w

    def _find_closest_point(
        self, x: float, y: float
    ) -> tuple[tuple[float, float], int]:
        """Find the closest path point to the current robot pose.

        Args:
            x: Estimated robot x coordinate [m].
            y: Estimated robot y coordinate [m].

        Returns:
            Tuple[float, float]: (x, y) coordinates of the closest path point [m].
            int: Index of the path point found.

        """
        # TODO: 4.9. Complete the function body (i.e., find closest_xy and closest_idx).
        closest_idx = np.argmin(
            np.linalg.norm(np.array(self._path) - np.array([x, y]), axis=1)
        )

        closest_xy = self._path[closest_idx]

        return closest_xy, closest_idx

    def _find_target_point(
        self, origin_xy: tuple[float, float], origin_idx: int
    ) -> tuple[float, float]:
        """Find the destination path point based on the lookahead distance.

        Args:
            origin_xy: Current location of the robot (x, y) [m].
            origin_idx: Index of the current path point.

        Returns:
            Tuple[float, float]: (x, y) coordinates of the target point [m].

        """
        # TODO: 4.10. Complete the function body with your code (i.e., determine target_xy).
        target_xy = (0.0, 0.0)
        target_idx = None
        candidate_points = np.array(self._path[origin_idx + 1 :])

        # When the robot is at the end of the path
        if origin_idx == len(self._path) - 1:
            return self._path[-1]

        distances = np.linalg.norm(candidate_points - np.array(origin_xy), axis=1)

        # Target is candidate point closest to distance l
        target_idx = np.argmin(np.abs(distances - self._lookahead_distance))
        target_xy = candidate_points[target_idx]

        return target_xy

    @property
    def path(self) -> list[tuple[float, float]]:
        """Path getter."""
        return self._path

    @path.setter
    def path(self, value: list[tuple[float, float]]) -> None:
        """Path setter."""
        self._path = value
