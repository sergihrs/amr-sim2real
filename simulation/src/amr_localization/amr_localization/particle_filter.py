import datetime
import math
import os
from typing import Iterator

import numpy as np
import pytz
from matplotlib import pyplot as plt
from scipy.stats import chi2
from sklearn.cluster import DBSCAN

from amr_localization.map import Map

N_LIDAR_RAYS = 16


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensor_range: float,
        particle_count: int,
        robot_center_clearance: float = 0.0,
        sigma_v: float = 0.25,
        sigma_w: float = 0.5,
        sigma_z: float = 0.25,
        logger=None,
        epsilon: float = 0.15,
        delta: float = 0.99,
        beta_ema: float = 0.5,
        bin_size: np.ndarray[float] = np.array([0.07, 0.07, 10.0]),
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sensor_range: Sensor measurement range [m].
            particle_count: Initial number of particles.
            robot_center_clearance: Minimum allowed distance from robot center to walls [m].
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].

        """
        self._dt: float = dt
        self._sensor_range: float = sensor_range
        self._missing_measurement: float = 1.25 * sensor_range
        self._robot_center_clearance: float = max(0.0, float(robot_center_clearance))
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0
        self._epsilon: float = epsilon
        self._delta: float = delta
        self._bin_size: np.ndarray[float] = bin_size
        self._beta_ema: float = beta_ema

        self._initial_particles: int = particle_count
        self.localized: bool = False
        self._n_low_weights: int = 0
        self._likelihood: float = None

        self._p_min = 100.0
        self._p_max = 200.0

        self._s_w_min = 0.6
        self._s_w_max = 0.9
        self._s_v_min = 0.2
        self._s_v_max = 0.25
        self._ray_angle_offsets = np.deg2rad(
            np.arange(N_LIDAR_RAYS, dtype=float) * (360.0 / N_LIDAR_RAYS)
        )

        self._map = Map(map_path, sensor_range, compiled_intersect=True, use_regions=True)
        self._particles = self._init_particles(self._initial_particles)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self._logger = logger

    def compute_pose(self) -> tuple[float, float, float]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        clusters = DBSCAN(eps=0.25, min_samples=10).fit(self._particles[:, :2])
        n_clusters = len(np.unique(clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)

        if n_clusters == 1:
            localized = True
            particles_4d = np.concatenate(
                (
                    self._particles[:, :2],
                    np.cos(self._particles[:, 2:3]),
                    np.sin(self._particles[:, 2:3]),
                ),
                axis=1,
            )

            # Standard deviation of the position
            centroid = np.mean(particles_4d[:, :2], axis=0)
            dists = np.linalg.norm(particles_4d[:, :2] - centroid, axis=1)
            sigma = np.mean(dists)

            if sigma > 0.1:
                self.localized = False
                return (float("inf"), float("inf"), float("inf"))

            pose_4d = np.mean(particles_4d, axis=0)

            pose = (
                float(pose_4d[0]),
                float(pose_4d[1]),
                float(math.atan2(pose_4d[3], pose_4d[2])),
            )

        self.localized = localized

        return pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            p: Particle.
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        m = len(self._particles)

        v_noise = np.random.normal(v, self._sigma_v, m)
        w_noise = np.random.normal(w, self._sigma_w, m)

        self._particles[:, 2] = (self._particles[:, 2] + w_noise * self._dt) % (2 * math.pi)
        self._particles[:, 0] += v_noise * np.cos(self._particles[:, 2]) * self._dt
        self._particles[:, 1] += v_noise * np.sin(self._particles[:, 2]) * self._dt

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles, using KLD resampling.

        Args:
            measurements: Sensor measurements [m].

        """
        filtered_measurements = self._prepare_measurements(measurements)

        weights = np.fromiter(
            (
                self._measurement_probability(filtered_measurements, particle)
                for particle in self._particles
            ),
            dtype=float,
            count=len(self._particles),
        )

        mean_weight = float(weights.mean())

        if self._likelihood is None:
            self._likelihood = mean_weight
        else:
            self._likelihood = (
                self._beta_ema * mean_weight + (1 - self._beta_ema) * self._likelihood
            )
        # self._logger.info(f"Mean weight: {mean_weight:.4f} | likelihood: {self._likelihood:.4f}")

        if self._likelihood < 0.005 and self._iteration > 3:
            self._logger.info("Mean weight is too low. Resampling")

            self._particles = self._init_particles(self._initial_particles)

            self._iteration = 0
            self._likelihood = None
            self.localized = False
            return

        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 1e-300:
            weights = np.full(len(self._particles), 1.0 / len(self._particles))
        else:
            weights /= weight_sum

        # multinomial resampling using KLD
        x_min, y_min, _, _ = self._map.bounds()
        new_particles = []
        bins = set()
        m = 1
        cumulative_weights = np.cumsum(weights)

        def draw_index() -> int:
            u = np.random.random()
            return int(np.searchsorted(cumulative_weights, u, side="right"))

        # Sample the first particle
        i = draw_index()
        # new_particles[i] += 1  # Add one to the particle count
        new_particles.append(self._particles[i])  # Add the particle

        # Get the bin the particle is in
        bin_index = (
            (self._particles[i][0] - x_min) // self._bin_size[0],
            (self._particles[i][1] - y_min) // self._bin_size[1],
            self._particles[i][2] // self._bin_size[2],
        )

        # Add the bin to the set of bins
        bins.add(bin_index)

        eps = self._epsilon if not self.localized else 2 * self._epsilon

        while m < (2 / eps) * chi2.ppf(self._delta, len(bins)):
            # Sample the next particle
            i = draw_index()

            # new_particles[i] += 1  # Add one to the particle count
            new_particles.append(self._particles[i])  # Add the particle

            # Get the bin the particle is in
            bin_index = (
                (self._particles[i][0] - x_min) // self._bin_size[0],
                (self._particles[i][1] - y_min) // self._bin_size[1],
                self._particles[i][2] // self._bin_size[2],
            )
            bins.add(bin_index)

            m += 1

        # Update the sigmas based on the number of particles
        m = np.clip(m, self._p_min, self._p_max)

        self._sigma_w = (m - self._p_min) / (self._p_max - self._p_min) * (
            self._s_w_max - self._s_w_min
        ) + self._s_w_min

        self._sigma_v = (m - self._p_min) / (self._p_max - self._p_min) * (
            self._s_v_max - self._s_v_min
        ) + self._s_v_min

        # self._particles = self._particles.repeat(new_particles, axis=0)
        self._particles = np.array(new_particles)

        if self._iteration == 0:
            # Add noise to the particles in the first resampling, so that the
            # particles can spread and some end close to the actual robot pose
            s_x = 0.03
            s_y = 0.03
            self._particles[:, 0] += np.random.normal(0, s_x, len(self._particles))
            self._particles[:, 1] += np.random.normal(0, s_y, len(self._particles))

        self._iteration += 1

    def plot(self, axes, orientation: bool = True):
        """Draws particles.

        Args:
            axes: Figure axes.
            orientation: Draw particle orientation.

        Returns:
            axes: Modified axes.

        """
        if orientation:
            dx = [math.cos(particle[2]) for particle in self._particles]
            dy = [math.sin(particle[2]) for particle in self._particles]
            axes.quiver(
                self._particles[:, 0],
                self._particles[:, 1],
                dx,
                dy,
                color="b",
                scale=15,
                scale_units="inches",
            )
        else:
            axes.plot(self._particles[:, 0], self._particles[:, 1], "bo", markersize=1)

        return axes

    def show(
        self,
        title: str = "",
        orientation: bool = True,
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays the current particle set on the map.

        Args:
            title: Plot title.
            orientation: Draw particle orientation.
            display: True to open a window to visualize the particle filter evolution in real-time.
                Time consuming. Does not work inside a container unless the screen is forwarded.
            block: True to stop program execution until the figure window is closed.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """

        figure = self._figure
        axes = self._axes
        axes.clear()

        axes = self._map.plot(axes)
        axes = self.plot(axes, orientation)

        axes.set_title(title + " (Iteration #" + str(self._iteration) + ")")
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "..", save_dir, self._timestamp)
            )
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            last_iter = len(os.listdir(save_path))

            file_name = str(last_iter).zfill(4) + " " + title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=float)

        map_boundaries = self._map.bounds_with_clearance(self._robot_center_clearance)
        if map_boundaries is None:
            raise ValueError(
                "No valid sampling region for configured robot_center_clearance="
                f"{self._robot_center_clearance:.3f} m"
            )

        x_min, y_min, x_max, y_max = map_boundaries

        particles[:, 2] = np.random.choice(
            [0, math.pi / 2, math.pi, 3 * math.pi / 2], particle_count
        )

        max_total_attempts = 500 * particle_count
        attempts = 0
        for i in range(particle_count):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            while not self._map.contains_with_clearance((x, y), self._robot_center_clearance):
                attempts += 1
                if attempts > max_total_attempts:
                    raise RuntimeError(
                        "Particle initialization exceeded maximum attempts. "
                        "Try reducing robot_center_clearance or reviewing the map geometry."
                    )
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)

            particles[i, 0] = x
            particles[i, 1] = y

        return particles

    def _sense(self, pose: tuple[float, float, float]) -> Iterator[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns: Predicted measurements [m].

        """
        z_hat: Iterator[float] = self._lidar_rays(
            pose, range(N_LIDAR_RAYS), degree_increment=360 / N_LIDAR_RAYS
        )

        z_hat = (self._map.check_collision(ray, compute_distance=True)[1] for ray in z_hat)

        return z_hat

    @staticmethod
    def _gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    @staticmethod
    def _log_gaussian(mu: float, sigma: float, x: float) -> float:
        """Computes the value of a Gaussian.

        Args:
            mu: Mean.
            sigma: Standard deviation.
            x: Variable.

        Returns:
            float: Gaussian value.

        """
        return -0.5 * ((x - mu) / sigma) ** 2  # - np.log(sigma * np.sqrt(2 * np.pi))

    def _prepare_measurements(self, measurements: list[float]) -> np.ndarray:
        """Downsamples and sanitizes LiDAR scan measurements once per sense step."""
        measurements_array = np.asarray(measurements, dtype=float)
        indices = (np.arange(N_LIDAR_RAYS) * len(measurements_array)) // N_LIDAR_RAYS
        sampled = measurements_array[indices]

        return np.nan_to_num(
            sampled,
            nan=self._missing_measurement,
            posinf=self._missing_measurement,
            neginf=self._missing_measurement,
        )

    def _measurement_probability(
        self, filtered_measurements: np.ndarray, pose: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        1.25 times the sensor range to perform the computation. This value has experimentally been
        proven valid to deal with missing measurements. Nevertheless, it might not be the optimal
        replacement value.

        Args:
            filtered_measurements: Filtered and downsampled sensor measurements [m].
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        x, y, theta = pose
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)
        ray_angles = theta + self._ray_angle_offsets

        particle_measurements = self._map.raycast_distances(
            (x_start, y_start),
            ray_angles,
            self._sensor_range,
            self._missing_measurement,
        )

        error = (particle_measurements - filtered_measurements) / self._sigma_z
        log_likelihood = -0.5 * float(np.dot(error, error))

        # Clamp to avoid underflow turning all weights into zeros.
        return float(np.exp(max(log_likelihood, -700.0)))

    def _lidar_rays(
        self,
        pose: tuple[float, float, float],
        indices: tuple[float],
        degree_increment: float = 1.5,
    ) -> Iterator[tuple[tuple[float, float], tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot pose (x, y, theta) in [m] and [rad].
            indices: Rays of interest in counterclockwise order. Index 0 corresponds to frontal ray.
            degree_increment: Angle difference between contiguous rays [degrees].

        Returns: Ray segments. Format:
                 [[(x0_start, y0_start), (x0_end, y0_end)],
                  [(x1_start, y1_start), (x1_end, y1_end)],
                  ...]

        """
        x, y, theta = pose

        # Convert sensor origin to world coordinates
        x_start = x - 0.035 * math.cos(theta)
        y_start = y - 0.035 * math.sin(theta)

        rays = (
            (
                (x_start, y_start),
                (
                    x_start + self._sensor_range * math.cos(theta + ray_angle),
                    y_start + self._sensor_range * math.sin(theta + ray_angle),
                ),
            )
            for ray_angle in (math.radians(degree_increment * index) for index in indices)
        )

        return rays
