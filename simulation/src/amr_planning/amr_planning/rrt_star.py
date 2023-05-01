import datetime
import heapq
import os
from typing import Generator

import numpy as np
import pytz
from matplotlib import pyplot as plt

from amr_planning.map import Map


class RRTStar:
    """Class to plan the optimal path to a given location using the RRT* algorithm."""

    def __init__(
        self,
        map_path: str,
        sensor_range: float,
        safety_distance: float = 0.0,
        exploration_distance: float = 0.25,
        neighbor_distance: float = 0.1,
        logger: None = None,
    ):
        """RRT* class initializer.

        Args:
            map_path: Path to the map of the environment.
            sensor_range: Sensor measurement range [m].
            safety_distance: Minimum separation distance the robot should keep with the walls [m].
            exploration_distance: Maximum distance between tree nodes [m].
            neighbor_distance: Reconnection radius [m].

        """
        self._exploration_distance: float = exploration_distance
        self._map: Map = Map(
            map_path,
            sensor_range,
            safety_distance,
            compiled_intersect=False,
            use_regions=False,
        )
        self._neighbor_distance: float = neighbor_distance

        self._figure, self._axes = plt.subplots(1, 1, figsize=(7, 7))
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        # List of nodes
        self._nodes: list[tuple[float, float]] = []
        self._ancestors: dict[tuple, tuple[tuple, float]] = {}

        self._logger = logger

    def compute_path(
        self, start: tuple[float, float], goal: tuple[float, float]
    ) -> list[tuple[float, float]]:
        """Computes the quasi-optimal path to a given goal location using the RRT* algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Destination in (x, y) format.

        Returns:
            Path to the destination. The first value corresponds to the initial location.

        """
        # TODO 4.3: Implement RRT*
        self._nodes = [start]
        self._ancestors = {start: (None, 0.0)}

        new_node = self._generate_node()
        dist_to_goal, best_node = float("inf"), None
        epochs_no_improvement = 0
        found_path = False
        while epochs_no_improvement < 500:
            self._reconnect_neighbors(new_node)
            new_node = self._generate_node()

            # Save a figure with the current tree
            # self.save_tree(start, goal, new_node)

            # If new_node to goal does not cross a wall
            if not self._map.crosses((new_node, goal)):
                node_dist_to_goal = np.linalg.norm(np.array(new_node) - np.array(goal))
                new_dist_to_goal = self._get_distance_to_start(new_node) + node_dist_to_goal
                if best_node:
                    best_dist_to_goal = self._get_distance_to_start(best_node) + dist_to_goal
                    new_dist_to_goal = min(new_dist_to_goal, best_dist_to_goal)

                # Node must be close to the goal
                if node_dist_to_goal > self._exploration_distance:
                    epochs_no_improvement += 1
                    continue

                # Check if the new node is closer to the goal
                if dist_to_goal - new_dist_to_goal > 3e-3:
                    found_path = True
                    dist_to_goal = new_dist_to_goal
                    best_node = new_node
                    epochs_no_improvement = 0

            if found_path:
                epochs_no_improvement += 1

        self._nodes.append(goal)
        self._ancestors[goal] = (
            best_node,
            np.linalg.norm(np.array(best_node) - np.array(goal)),
        )
        path = self._reconstruct_path(start, goal, self._ancestors)

        return path

    def _generate_node(self) -> tuple[float, float]:
        """Generates a node in free space of the safe map.

        Returns:
            Coordinates of the generated node.

        """

        # TODO: 4.1. Complete the function body with your code (i.e., replace the pass statement).
        def valid_point(point: tuple[float, float]):
            # 1. Distance from tree to point
            dist = np.linalg.norm(node_array - point, axis=1)
            nearest = np.argmin(dist)

            # 2. Closen the point to exploration distance
            v = np.array(point) - np.array(self._nodes[nearest])
            point = self._nodes[nearest] + min(
                dist[nearest], self._exploration_distance
            ) * v / np.linalg.norm(v)

            # 3. Check if the point is contained in the map
            contained = self._map.contains(point)
            if not contained:
                return None, None, None, False

            # 4. Check if the point crosses a wall
            collision = self._map.crosses((self._nodes[nearest], point))
            return point, nearest, dist[nearest], contained and not collision

        node_array = np.array(self._nodes)
        x_min, y_min, x_max, y_max = self._map.bounds()

        # generate random point in the bounding box of the map
        new_point = np.random.uniform((x_min, y_min), (x_max, y_max))
        new_point, nearest, distance, valid = valid_point(new_point)
        while not valid:
            new_point = np.random.uniform((x_min, y_min), (x_max, y_max))
            new_point, nearest, distance, valid = valid_point(new_point)

        # Add the new node to the list of nodes
        new_node = tuple(new_point)
        self._nodes.append(new_node)
        self._ancestors[new_node] = (self._nodes[nearest], distance)

        return new_node

    def _get_distance_to_start(self, node: tuple[float, float]) -> float:
        """Computes the distance from a node to the start location.

        Args:
            node: Coordinates of the node.

        Returns: Distance from the node to the start location.

        """
        distance: float = 0.0
        current_node = node
        while self._ancestors[current_node][0] is not None:
            distance += self._ancestors[current_node][1]
            current_node = self._ancestors[current_node][0]
        return distance

    def _reconnect_neighbors(self, node: tuple[float, float]):
        """Reconnects the proximity of a newly added node.

        Args:
            node: Coordinates of the node.

        """
        # TODO: 4.2. Complete the function body with your code (i.e., replace the pass statement).
        other_nodes = self._nodes.copy()
        other_nodes.remove(node)
        dist = np.linalg.norm(np.array(other_nodes) - np.array(node), axis=1)
        best_neigh_idx = np.argmin(dist)
        neighbors_indexes = np.where(dist < self._neighbor_distance)[0]

        # 1. Connect the new node to the best neighbor
        self._ancestors[node] = (other_nodes[best_neigh_idx], dist[best_neigh_idx])

        # 2. Reconnect the neighbors of the new node
        for neigh_idx in neighbors_indexes:
            if self._map.crosses((other_nodes[neigh_idx], node)):
                continue
            cost = self._get_distance_to_start(node) + dist[neigh_idx]
            if cost < self._get_distance_to_start(other_nodes[neigh_idx]):
                self._ancestors[node] = (other_nodes[neigh_idx], dist[neigh_idx])
        return None

    def _reconstruct_path(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        ancestors,
    ) -> list[tuple[float, float]]:
        """Computes the path from the start to the goal given the ancestors of a search algorithm.

        Args:
            start: Initial location in (x, y) format.
            goal: Goal location in (x, y) format.
            ancestors: Collection of ancestors for every node.

        Returns: Path to the goal (start location first) in (x, y) format.

        """
        # TODO 4.4: Complete the function body with your code (i.e., replace the pass statement).
        path = []
        n = goal
        while n != start:
            path.append(n)
            n = ancestors[n][0]
        path.append(start)
        path.reverse()
        return path

    def save_tree(self, start, goal, new_node):
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        self._map.plot(ax)
        for node in self._nodes:
            if self._ancestors[node][0] is not None:
                ax.plot(
                    [node[0], self._ancestors[node][0][0]],
                    [node[1], self._ancestors[node][0][1]],
                    "b",
                )
            ax.plot(node[0], node[1], "bo", markersize=4)
        ax.plot(start[0], start[1], "rs", markersize=7)
        ax.plot(goal[0], goal[1], "g*", markersize=12)
        ax.plot(new_node[0], new_node[1], "bo", markersize=4)
        ax.set_title("RRT*")

        # Save the figure
        save_path = os.path.join(os.path.dirname(__file__), "..", "images", "rrt_star")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        # Save current timestamp
        timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        file_name = f"{timestamp} RRT*.png"

        fig.savefig(os.path.join(save_path, file_name))
        plt.close(fig)

    @staticmethod
    def plot(
        axes,
        path: list[tuple[float, float]],
        smoothed_path: list[tuple[float, float]] = (),
    ):
        """Draws a path.

        Args:
            axes: Figure axes.
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).

        Returns:
            axes: Modified axes.

        """
        x_val = [x[0] for x in path]
        y_val = [x[1] for x in path]

        axes.plot(x_val, y_val)  # Plot the path
        axes.plot(
            x_val[1:-1], y_val[1:-1], "bo", markersize=4
        )  # Draw blue circles in every intermediate cell

        if smoothed_path:
            x_val = [x[0] for x in smoothed_path]
            y_val = [x[1] for x in smoothed_path]

            axes.plot(x_val, y_val, "y")  # Plot the path
            axes.plot(
                x_val[1:-1], y_val[1:-1], "yo", markersize=4
            )  # Draw yellow circles in every intermediate cell

        axes.plot(x_val[0], y_val[0], "rs", markersize=7)  # Draw a red square at the start location
        axes.plot(
            x_val[-1], y_val[-1], "g*", markersize=12
        )  # Draw a green star at the goal location

        return axes

    def show(
        self,
        path,
        smoothed_path=(),
        title: str = "Path",
        display: bool = False,
        block: bool = False,
        save_figure: bool = False,
        save_dir: str = "images",
    ):
        """Displays a given path on the map.

        Args:
            path: Path (start location first).
            smoothed_path: Smoothed path (start location first).
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
        axes = self.plot(axes, path, smoothed_path)

        axes.set_title(title)
        figure.tight_layout()  # Reduce white margins

        if display:
            plt.show(block=block)
            plt.pause(0.001)  # Wait for 1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.join(os.path.dirname(__file__), "..", save_dir)

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = f"{self._timestamp} {title.lower()}.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)


class AStar:
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
        self._timestamp = datetime.datetime.now(pytz.timezone("Europe/Madrid")).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        self._logger = logger

        self._n = 80
        self._m = 80

        self._min_distance = 0.1

        self.add_obstacles()

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

        pq = [(0.0, start)]

        parents = {start: None}
        visited = set()

        v = None
        while pq and v != goal:
            f_score, v = heapq.heappop(pq)
            # Ignore stale queue entries that were superseded by a better score.
            if f_score > min_distances[v] + self._h_distances[v[0], v[1]]:
                continue

            visited.add(v)

            for w in self.neighbors(v):
                if w not in visited:
                    new_distance = min_distances[v] + self.distance_between(v, w)

                    if new_distance < min_distances[w]:
                        min_distances[w] = new_distance
                        parents[w] = v
                        heapq.heappush(
                            pq,
                            (new_distance + self._h_distances[w[0], w[1]], w),
                        )

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
        x_min, y_min, x_max, y_max = self._map.bounds()

        self.lx = (x_max - x_min) / self._m
        self.ly = (y_max - y_min) / self._n

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
        x_min, y_min, x_max, y_max = self._map.bounds()

        # Get closest node to start and goal
        start = (
            int((start[1] - y_min - self.ly / 2) / self.ly),
            int((start[0] - x_min - self.lx / 2) / self.lx),
        )
        goal = (
            int((goal[1] - y_min - self.ly / 2) / self.ly),
            int((goal[0] - x_min - self.lx / 2) / self.lx),
        )

        non_obstacle_indices = np.argwhere(self._inside)
        closest_to_goal = np.argmin(
            np.linalg.norm(non_obstacle_indices - np.array([goal[0], goal[1]]), axis=1)
        )
        closest_to_start = np.argmin(
            np.linalg.norm(non_obstacle_indices - np.array([start[0], start[1]]), axis=1)
        )

        start = tuple(non_obstacle_indices[closest_to_start])
        goal = tuple(non_obstacle_indices[closest_to_goal])

        self._h_distances = np.array(
            [
                [np.linalg.norm([x - goal[0], y - goal[1]]) for y in range(self._m)]
                for x in range(self._n)
            ]
        )

        path = self.find_path(start, goal)

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
                np.linalg.norm([node[0] - processed_path[-1][0], node[1] - processed_path[-1][1]])
                > self._min_distance
            ):
                processed_path.append(node)

        return processed_path

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
