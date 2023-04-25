import datetime
import math
import numpy as np
import os
import pytz
from time import perf_counter

from amr_localization.map import Map
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from scipy.stats import chi2

from typing import Iterator

# Execution time: 325.736 s (0.931 s/step) | Simulated time: 17.500 s (350 steps)
# Execution time: 299.149 s (0.882 s/step) | Simulated time: 16.950 s (339 steps)
# Execution time: 310.824 s (0.893 s/step) | Simulated time: 17.400 s (348 steps)

# Execution time: 320.572 s (0.946 s/step) | Simulated time: 16.950 s (339 steps)
# Execution time: 336.082 s (0.988 s/step) | Simulated time: 17.000 s (340 steps)

# Execution time: 331.518 s (0.964 s/step) | Simulated time: 17.200 s (344 steps)

# Execution time: 307.204 s (0.912 s/step) | Simulated time: 16.850 s (337 steps)

# Esto ha sido de volver a lista frente a máscara
# Execution time: 259.231 s (0.741 s/step) | Simulated time: 17.500 s (350 steps)
# WTF mismo código, por que hay tanta varianza?
# Execution time: 357.415 s (1.033 s/step) | Simulated time: 17.300 s (346 steps)
# Flipo
# Execution time: 335.355 s (0.981 s/step) | Simulated time: 17.100 s (342 steps)

# Metiendo 2 moves por cada sense cuando localized
# Execution time: 283.434 s (0.796 s/step) | Simulated time: 17.800 s (356 steps)
# 5 moves
# Execution time: 283.130 s (0.795 s/step) | Simulated time: 17.800 s (356 steps)

N_LIDAR_RAYS = 16


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensor_range: float,
        particle_count: int,
        sigma_v: float = 0.25,
        sigma_w: float = 0.5,
        sigma_z: float = 0.4,
        logger=None,
        epsilon: float = 0.2,
        delta: float = 0.99,
        bin_size: np.ndarray = np.array([0.07, 0.07, 10.0]),
    ):
        """Particle filter class initializer.

        Args:
            dt: Sampling period [s].
            map_path: Path to the map of the environment.
            sensor_range: Sensor measurement range [m].
            particle_count: Initial number of particles.
            sigma_v: Standard deviation of the linear velocity [m/s].
            sigma_w: Standard deviation of the angular velocity [rad/s].
            sigma_z: Standard deviation of the measurements [m].

        """
        self._dt: float = dt
        self._sensor_range: float = sensor_range
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0
        self._epsilon: float = epsilon
        self._delta: float = delta
        self._bin_size: np.ndarray = bin_size

        self._p_min = 50.0
        self._p_max = 300.0

        self._s_w_min = 0.5
        self._s_w_max = 0.9
        self._s_v_min = 0.1
        self._s_v_max = 0.25

        self._map = Map(
            map_path, sensor_range, compiled_intersect=True, use_regions=False
        )
        self._particles = self._init_particles(particle_count)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(
            pytz.timezone("Europe/Madrid")
        ).strftime("%Y-%m-%d_%H-%M-%S")

        self._logger = logger

        self._last_time = None

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        clusters = DBSCAN(eps=0.25, min_samples=10).fit(self._particles[:, :2])
        n_clusters = len(np.unique(clusters.labels_)) - (
            1 if -1 in clusters.labels_ else 0
        )

        self._logger.info(f"Number of clusters: {n_clusters}")

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
            # self._logger.info(f"STD of the position: {sigma:.3f} m")

            # if sigma > 0.5:
            #     localized = False

            pose_4d = np.mean(particles_4d, axis=0)

            pose = (
                float(pose_4d[0]),
                float(pose_4d[1]),
                float(math.atan2(pose_4d[3], pose_4d[2])),
            )

        return localized, pose

    def substract_stop_time(self, time: float):
        if self._last_time is not None:
            self._last_time += time

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            p: Particle.
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1
        m = len(self._particles)

        v_noise = np.random.normal(v, self._sigma_v, m)
        w_noise = np.random.normal(w, self._sigma_w, m)

        if self._last_time is None:
            self._last_time = perf_counter()
            return 
    
        dt = perf_counter() - self._last_time
        self._last_time = perf_counter()

        self._logger.info(f"Time: {dt:.3f} s")
    

        self._particles[:, 2] = (self._particles[:, 2] + w_noise * dt) % (
            2 * math.pi
        )
        self._particles[:, 0] += v_noise * np.cos(self._particles[:, 2]) * dt
        self._particles[:, 1] += v_noise * np.sin(self._particles[:, 2]) * dt

        # TODO no checkear collisions y vectorizar
        # for i, particle in enumerate(self._particles):
        #     x, y, theta = particle
        #     x_initial, y_initial, _ = particle

        #     v_noisy = np.random.normal(v, self._sigma_v)
        #     w_noisy = np.random.normal(w, self._sigma_w)

        #     theta = (theta + w_noisy * self._dt) % (2 * math.pi)
        #     x += v_noisy * math.cos(theta) * self._dt
        #     y += v_noisy * math.sin(theta) * self._dt

        #     intersection, _ = self._map.check_collision(
        #         ((x_initial, y_initial), (x, y))
        #     )
        #     if intersection:
        #         x, y = intersection

        #     self._particles[i] = (x, y, theta)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles, using KLD resampling.

        Args:
            measurements: Sensor measurements [m].

        """
        start_time = perf_counter()
        weights = np.fromiter(
            (
                self._measurement_probability(measurements, particle)
                for particle in self._particles
            ),
            dtype=float,
            count=len(self._particles),
        )
        weights /= weights.sum()
        end_time = perf_counter()
        self._logger.info(
            f"Measurement probability time: {end_time - start_time:.3f} s"
        )

        # multinomial resampling using KLD
        x_min, y_min, _, _ = self._map.bounds()
        # new_particles = np.zeros(len(self._particles), dtype=np.uint16)
        new_particles = []
        bins = set()
        m = 1
        particle_count = len(self._particles)

        # Sample the first particle
        i = np.random.choice(particle_count, p=weights)
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

        while m < (2 / self._epsilon) * chi2.ppf(self._delta, len(bins)):
            # Sample the next particle
            i = np.random.choice(particle_count, p=weights)

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

        if self._iteration == 1:
            # Add noise to the particles in the first resampling, so that the
            # particles can spread and some end close to the actual robot pose
            s_x = 0.03
            s_y = 0.03
            self._particles[:, 0] += np.random.normal(0, s_x, len(self._particles))
            self._particles[:, 1] += np.random.normal(0, s_y, len(self._particles))

        self._iteration += 1
        self._logger.info(f"Resampled {len(self._particles)} particles.")

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

            file_name = str(self._iteration).zfill(4) + " " + title.lower() + ".png"
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

        map_boundaries = self._map.bounds()
        x_min, y_min, x_max, y_max = map_boundaries

        particles[:, 2] = np.random.choice(
            [0, math.pi / 2, math.pi, 3 * math.pi / 2], particle_count
        )
        for i in range(particle_count):

            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            while not self._map.contains((x, y)):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)

            particles[i, 0] = x
            particles[i, 1] = y

        return particles

    def _sense(self, pose: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns: Predicted measurements [m].

        """
        z_hat: Iterator[float] = self._lidar_rays(
            pose, range(N_LIDAR_RAYS), degree_increment=360 / N_LIDAR_RAYS
        )

        z_hat = [
            self._map.check_collision(ray, compute_distance=True)[1] for ray in z_hat
        ]

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

    def _measurement_probability(
        self, measurements: list[float], pose: tuple[float, float, float]
    ) -> float:
        """Computes the probability of a set of measurements given a particle's pose.

        If a measurement is unavailable (usually because it is out of range), it is replaced with
        1.25 times the sensor range to perform the computation. This value has experimentally been
        proven valid to deal with missing measurements. Nevertheless, it might not be the optimal
        replacement value.

        Args:
            measurements: Sensor measurements [m].
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns:
            float: Probability.

        """
        particle_measurements = np.array(self._sense(pose))

        filtered_measurements = np.array([
            measurements[i * len(measurements) // N_LIDAR_RAYS]
            for i in range(N_LIDAR_RAYS)
        ])

        nan_indexes_robot = np.isnan(filtered_measurements)
        nan_indexes_particle = np.isnan(particle_measurements)
        nan_indexes = nan_indexes_robot | nan_indexes_particle

        particle_measurements = particle_measurements[~nan_indexes]
        filtered_measurements = filtered_measurements[~nan_indexes]

        filtered_measurements = np.nan_to_num(
            filtered_measurements,
            nan=1.25 * self._sensor_range,
            posinf=1.25 * self._sensor_range,
            neginf=1.25 * self._sensor_range,
        )
        particle_measurements = np.nan_to_num(
            particle_measurements,
            nan=1.25 * self._sensor_range,
            posinf=1.25 * self._sensor_range,
            neginf=1.25 * self._sensor_range,
        )

        log_likelihood = (
            -0.5
            * ((particle_measurements - filtered_measurements) / self._sigma_z) ** 2
        ).sum()

        return np.exp(log_likelihood)

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
            for ray_angle in (
                math.radians(degree_increment * index) for index in indices
            )
        )

        return rays
