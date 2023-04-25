import datetime
import math
import numpy as np
import os
import pytz

# import random

from amr_localization.map import Map
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

N_LIDAR_RAYS = 16


class ParticleFilter:
    """Particle filter implementation."""

    def __init__(
        self,
        dt: float,
        map_path: str,
        sensor_range: float,
        particle_count: int,
        sigma_v: float = 0.15,
        sigma_w: float = 0.75,
        sigma_z: float = 0.25,
        logger=None,
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
        self._initial_particle_count: int = particle_count
        self._particle_count: int = particle_count
        self._sensor_range: float = sensor_range
        self._sigma_v: float = sigma_v
        self._sigma_w: float = sigma_w
        self._sigma_z: float = sigma_z
        self._iteration: int = 0

        self._map = Map(
            map_path, sensor_range, compiled_intersect=True, use_regions=False
        )
        self._particles = self._init_particles(particle_count)
        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(
            pytz.timezone("Europe/Madrid")
        ).strftime("%Y-%m-%d_%H-%M-%S")

        self._logger = logger
        self._localized = False
        self._running_avg_likelihood = 0.0

    def compute_pose(self) -> tuple[bool, tuple[float, float, float]]:
        """Computes the pose estimate when the particles form a single DBSCAN cluster.

        Adapts the amount of particles depending on the number of clusters during localization.
        100 particles are kept for pose tracking.

        Returns:
            localized: True if the pose estimate is valid.
            pose: Robot pose estimate (x, y, theta) [m, m, rad].

        """
        localized: bool = False
        pose: tuple[float, float, float] = (float("inf"), float("inf"), float("inf"))

        # TODO: 3.10. Complete the missing function body with your code.
        clusters = DBSCAN(eps=0.25, min_samples=10).fit(self._particles[:, :2])
        # n_clusters = len(np.unique(clusters.labels_)) - (
        #     1 if -1 in clusters.labels_ else 0
        # )
        n_clusters = len(np.unique(clusters.labels_))

        # self._logger.info(f"Number of clusters: {n_clusters}")

        if n_clusters == 1:
            localized = True
            particles_4d = np.array(
                [
                    [
                        particle[0],
                        particle[1],
                        math.cos(particle[2]),
                        math.sin(particle[2]),
                    ]
                    for particle in self._particles
                ]
            )

            # Standard deviation of the position
            centroid = np.mean(particles_4d[:, :2], axis=0)
            dists = np.linalg.norm(particles_4d[:, :2] - centroid, axis=1)
            sigma = np.mean(dists)
            # self._logger.info(f"STD of the position: {sigma:.3f} m")

            MAX_CLUSTER_STD = 0.1

            if sigma > MAX_CLUSTER_STD:
                localized = False

            pose_4d = np.mean(particles_4d, axis=0)

            pose = (
                float(pose_4d[0]),
                float(pose_4d[1]),
                float(math.atan2(pose_4d[3], pose_4d[2])),
            )

        elif n_clusters > 1:
            # Reduce the number of particles proportional to the number of clusters
            PARTICLES_PER_CLUSTER = 200
            self._particle_count = min(
                n_clusters * PARTICLES_PER_CLUSTER, self._initial_particle_count
            )

        self._localized = localized

        return localized, pose

    def move(self, v: float, w: float) -> None:
        """Performs a motion update on the particles.

        Args:
            v: Linear velocity [m].
            w: Angular velocity [rad/s].

        """
        self._iteration += 1

        # TODO: 3.5. Complete the function body with your code (i.e., replace the pass statement).
        for i, particle in enumerate(self._particles):
            x, y, theta = particle
            x_initial, y_initial, _ = particle

            v_noisy = np.random.normal(v, self._sigma_v)
            w_noisy = np.random.normal(w, self._sigma_w)

            x += v_noisy * math.cos(theta + w_noisy * self._dt) * self._dt
            y += v_noisy * math.sin(theta + w_noisy * self._dt) * self._dt
            theta = (theta + w_noisy * self._dt) % (2 * math.pi)

            intersection, _ = self._map.check_collision(
                [(x_initial, y_initial), (x, y)]
            )
            if intersection:
                x, y = intersection

            self._particles[i] = (x, y, theta)

    def resample(self, measurements: list[float]) -> None:
        """Samples a new set of particles.

        Args:
            measurements: Sensor measurements [m].

        """
        # TODO: 3.9. Complete the function body with your code (i.e., replace the pass statement).
        likelihoods = np.array(
            [
                self._measurement_probability(measurements, particle)
                for particle in self._particles
            ]
        )
        # Check average likelihood
        avg_likelihood = np.mean(likelihoods)
        BETA = 0.4
        self._running_avg_likelihood = (
            1 - BETA
        ) * self._running_avg_likelihood + BETA * avg_likelihood
        self._logger.info(
            f"Average likelihood: {avg_likelihood:.3f} ({self._running_avg_likelihood:.3f})"
        )

        if self._localized and self._running_avg_likelihood < 0.01:
            self._logger.warning(
                f"RESAMPLING due to low average likelihood: {self._running_avg_likelihood:.3f}"
            )
            self._particles = self._init_particles(self._particle_count)
            self._localized = False
            return None

        # Normalize weights
        weights = likelihoods / np.sum(likelihoods)

        # Systematic resampling
        new_particles = np.empty((self._particle_count, 3), dtype=object)

        r = np.random.uniform(0, 1 / self._particle_count)
        c = weights[0]

        i = 0
        for m in range(self._particle_count):
            U = r + m / self._particle_count
            while U > c:
                i += 1
                c += weights[i]

            new_particles[m] = self._particles[i]

        self._particles = new_particles

    def _init_particles(self, particle_count: int) -> np.ndarray:
        """Draws N random valid particles.

        The particles are guaranteed to be inside the map and
        can only have the following orientations [0, pi/2, pi, 3*pi/2].

        Args:
            particle_count: Number of particles.

        Returns: A NumPy array of tuples (x, y, theta) [m, m, rad].

        """
        particles = np.empty((particle_count, 3), dtype=object)

        # TODO: 3.4. Complete the missing function body with your code.
        map_boundaries = self._map.bounds()
        x_min, y_min, x_max, y_max = map_boundaries

        for i in range(particle_count):
            theta = np.random.choice([0, math.pi / 2, math.pi, 3 * math.pi / 2])

            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)

            while not self._map.contains((x, y)):
                x = np.random.uniform(x_min, x_max)
                y = np.random.uniform(y_min, y_max)
            particles[i] = (x, y, theta)

        return particles

    def _sense(self, pose: tuple[float, float, float]) -> list[float]:
        """Obtains the predicted measurement of every sensor given the robot's pose.

        Args:
            pose: Particle pose (x, y, theta) [m, m, rad].

        Returns: List of predicted measurements.

        """
        # TODO: 3.6. Complete the missing function body with your code.
        z_hat: list[float] = self._lidar_rays(
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
        # TODO: 3.7. Complete the function body (i.e., replace the code below).
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
        return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))

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
        # TODO: 3.8. Complete the missing function body with your code.
        particle_measurements = self._sense(pose)

        def filter_measurements(x):
            return (
                x
                if not math.isnan(x) and not math.isinf(x)
                else 1.25 * self._sensor_range
            )

        measurements = [
            measurements[i * len(measurements) // N_LIDAR_RAYS]
            for i in range(N_LIDAR_RAYS)
        ]
        measurements = list(map(filter_measurements, measurements))
        particle_measurements = list(map(filter_measurements, particle_measurements))

        log_likelihood = sum(
            self._log_gaussian(z, self._sigma_z, z_hat)
            for z, z_hat in zip(measurements, particle_measurements)
        )

        return np.exp(log_likelihood)

    def _lidar_rays(
        self,
        pose: tuple[float, float, float],
        indices: tuple[float],
        degree_increment: float = 1.5,
    ) -> list[list[tuple[float, float]]]:
        """Determines the simulated LiDAR ray segments for a given robot pose.

        Args:
            pose: Robot or particle pose (x, y, theta) in [m] and [rad].
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

        rays = []

        for index in indices:
            ray_angle = math.radians(degree_increment * index)
            x_end = x_start + self._sensor_range * math.cos(theta + ray_angle)
            y_end = y_start + self._sensor_range * math.sin(theta + ray_angle)
            rays.append([(x_start, y_start), (x_end, y_end)])

        return rays

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
