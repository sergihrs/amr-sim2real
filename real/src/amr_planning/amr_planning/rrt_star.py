import datetime
from time import perf_counter
import numpy as np
import os
import pytz

from heapdict import heapdict
from astar import AStar as AStarFromLibrary

from typing import Generator

from amr_planning.map import Map
from matplotlib import pyplot as plt


class AStar(AStarFromLibrary):
    """Class to plan the optimal path to a given location using the A* algorithm."""

    def __init__(
        self,
        map_path: str,
        sensor_range: float,
        safety_distance: float = 0.0,
        logger: None = None,
    ):
        """A* class initializer"""

        self._map: Map = Map(
            map_path,
            sensor_range,
            safety_distance,
            compiled_intersect=False,
            use_regions=False,
        )

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(
            pytz.timezone("Europe/Madrid")
        ).strftime("%Y-%m-%d_%H-%M-%S")

        self._logger = logger

        self._n = 50
        self._m = 50

        self._min_distance = 0.15

    def neighbors(self, node: tuple) -> Generator:
        """Returns the neighbors of a given node.

        Args:
            node: Node in (x, y) format.

        Yields:
            Neighbors of the given node.

        """
        x, y = node

        if x > 0:
            if self._inside[x - 1, y]:
                yield (x - 1, y)

            if y > 0:
                if self._inside[x - 1, y - 1]:
                    yield (x - 1, y - 1)

            if y < self._m - 1:
                if self._inside[x - 1, y + 1]:
                    yield (x - 1, y + 1)

        if x < self._n - 1:
            if self._inside[x + 1, y]:
                yield (x + 1, y)

            if y > 0:
                if self._inside[x + 1, y - 1]:
                    yield (x + 1, y - 1)

            if y < self._m - 1:
                if self._inside[x + 1, y + 1]:
                    yield (x + 1, y + 1)

        if y > 0:
            if self._inside[x, y - 1]:
                yield (x, y - 1)

        if y < self._m - 1:
            if self._inside[x, y + 1]:
                yield (x, y + 1)

    def distance_between(self, node1: tuple, node2: tuple) -> float:
        """Computes the distance between two nodes.

        Args:
            node1: First node in (x, y) format.
            node2: Second node in (x, y) format.

        Returns:
            Distance between the two nodes.

        """
        return 1.0 if node1[0] == node2[0] or node1[1] == node2[1] else np.sqrt(2)

    def heuristic_cost_estimate(self, start: tuple, goal: tuple) -> float:
        """Computes the heuristic cost estimate between two nodes.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Heuristic cost estimate between the two nodes.

        """
        return self._h_distances[start[0], start[1]]

    def find_path(self, start: object, goal: object) -> list[tuple]:

        min_distances = {(x, y): np.inf for x in range(self._n) for y in range(self._m)}
        min_distances[start] = 0

        pq = heapdict()
        pq[start] = 0

        parents = {start: None}
        visited = set()

        v = None
        while pq and v != goal:
            v, _ = pq.popitem()
            visited.add(v)

            for w in self.neighbors(v):
                if w not in visited:
                    new_distance = min_distances[v] + self.distance_between(v, w)

                    if new_distance < min_distances[w]:
                        min_distances[w] = new_distance
                        parents[w] = v
                        pq[w] = new_distance + self._h_distances[w[0], w[1]]

        if v != goal:
            return []

        path = [v]
        while v != start:
            v = parents[v]
            path.append(v)

        return path[::-1]

    def add_obstacles(self) -> np.ndarray:
        """
        Maps the weighted grid to the map.
        """
        x_min, y_min, _, _ = self._map.bounds()
        self._inside = np.ones((self._n, self._m))

        for i in range(self._n):
            for j in range(self._m):
                x = x_min + j * self.lx + self.lx / 2
                y = y_min + i * self.ly + self.ly / 2
                if not self._map.contains((x, y)):
                    self._inside[i, j] = 0

    def compute_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Computes the quasi-optimal path to a given goal location using the A* algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        self._logger.info(f"Computing path from {start} to {goal}...")
        start_time = perf_counter()

        x_min, y_min, x_max, y_max = self._map.bounds()

        self.lx = (x_max - x_min) / self._m
        self.ly = (y_max - y_min) / self._n

        # Get closest node to start and goal
        start = (
            int((start[1] - y_min - self.ly / 2) / self.ly),
            int((start[0] - x_min - self.lx / 2) / self.lx),
        )
        goal = (
            int((goal[1] - y_min - self.ly / 2) / self.ly),
            int((goal[0] - x_min - self.lx / 2) / self.lx),
        )

        self.add_obstacles()

        non_obstacle_indices = np.argwhere(self._inside)
        closest_to_goal = np.argmin(
            np.linalg.norm(
                non_obstacle_indices - np.array([goal[0], goal[1]]), axis=1
            )
        )
        closest_to_start = np.argmin(
            np.linalg.norm(
                non_obstacle_indices - np.array([start[0], start[1]]), axis=1
            )
        )

        start = tuple(non_obstacle_indices[closest_to_start])
        goal = tuple(non_obstacle_indices[closest_to_goal])

        self._logger.info(f"Start cell: {start}, Goal cell: {goal}!!!!!!")

        self._h_distances = np.array(
            [
                [np.linalg.norm([x - goal[0], y - goal[1]]) for y in range(self._m)]
                for x in range(self._n)
            ]
        )

        path = self.astar(start, goal)

        scaled_path = [
            (
                x_min + (x + 0.5) * self.lx,
                y_min + (y + 0.5) * self.ly,
            )
            for y, x in path
        ]

        processed_path = [scaled_path[0]]

        for node in scaled_path[1:]:
            if (
                np.linalg.norm(
                    [node[0] - processed_path[-1][0], node[1] - processed_path[-1][1]]
                )
                > self._min_distance
            ):
                processed_path.append(node)

        self._logger.info(f"Path computed in {perf_counter() - start_time}.")

        return processed_path

    def smooth_path(
        self,
        path,
        data_weight: float = 0.1,
        smooth_weight: float = 0.1,
        tolerance: float = 1e-6,
    ) -> list[tuple[float, float]]:
        """Computes a smooth trajectory using gradient descent.

        Args:
            path: Non-smoothed path to the goal (start location first).
            data_weight: The larger, the more similar the output will be to the original path.
            smooth_weight: The larger, the smoother the output path will be.
            tolerance: The algorithm will stop when after an iteration the smoothed path changes
                       less than this value.

        Returns: Smoothed path (initial location first) in (x, y) format.

        """
        smoothed_path = []

        # TODO 4.5: Complete the function body with your code.
        smoothed_path = np.array(path, dtype=float)
        change = tolerance

        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                smoothed_path_i = smoothed_path[i]
                gradient = data_weight * (path[i] - smoothed_path_i) + smooth_weight * (
                    smoothed_path[i - 1] - 2 * smoothed_path_i + smoothed_path[i + 1]
                )
                # Check if point can be moved
                if self._map.crosses((path[i], smoothed_path_i + gradient)):
                    continue
                change += np.linalg.norm(gradient)
                smoothed_path[i] += gradient

        return list(map(tuple, smoothed_path))

    def show(
        self,
        path: np.ndarray,
        smoothed_path: list[tuple[float, float]] = (),
        title: str = "Path",
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays a given path on the map.

        Args:
            path: Path (start location first).
            title: Plot title.
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

        path = np.array(path)
        x_val = path[:, 0]
        y_val = path[:, 1]

        axes.plot(x_val, y_val)
        axes.plot(x_val[1:-1], y_val[1:-1], "bo", markersize=4)

        axes.plot(x_val[0], y_val[0], "rs", markersize=7)
        axes.plot(x_val[-1], y_val[-1], "g*", markersize=12)

        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, "y")  # Plot the path
            axes.plot(
                x_val[1:-1], y_val[1:-1], "yo", markersize=4
            )  # Draw yellow circles in every intermediate cell

        axes.set_title(title)
        figure.tight_layout()

        if display:
            plt.show(block=block)
            plt.pause(0.001)

        if save_figure:
            save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = f"{self._timestamp} {title.lower()}.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)
