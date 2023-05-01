import ctypes as ct
import json
import math
import os
import platform
from typing import List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import LineString, Polygon
from shapely.prepared import prep


class Map:
    """Class to perform operations on metric maps."""

    def __init__(
        self,
        json_file: str,
        sensor_range: float,
        use_regions: bool = True,
        compiled_intersect: bool = True,
    ):
        """Map class initializer.

        Args:
            json_file: Coordinates of the external boundary and the internal hole vertices.
            sensor_range: Sensor measurement range [m].
            use_regions: Split the map in regions to reduce the number of comparisons.
            compiled_intersect: Use compiled intersect library for increased performance.

        """
        # Load map from JSON file
        with open(json_file, "r") as read_file:
            data = json.load(read_file)

        boundary = data["metric"]["boundary"]
        holes = data["metric"]["holes"]

        # Create a polygon map
        self._map_polygon = Polygon(boundary, holes=holes)

        # Create a segment map
        self._map_segments = []

        for i in range(len(boundary) - 1):
            self._map_segments.append([boundary[i], boundary[i + 1]])

        self._map_segments.append([boundary[-1], boundary[0]])

        for hole in holes:
            for i in range(len(hole) - 1):
                self._map_segments.append([hole[i], hole[i + 1]])

            self._map_segments.append([hole[-1], hole[0]])

        # Create a grid map
        try:
            map_size = data["grid"]["size"]
            obstacles = data["grid"]["obstacles"]

            self._grid_map = np.zeros(map_size, np.int8)

            for obstacle in obstacles:
                self._grid_map[tuple(obstacle)] = 1
        except KeyError:
            self._grid_map = None

        # Performance optimizations
        self._sensor_range = sensor_range
        self._intersect = self._init_intersect() if compiled_intersect else None
        self._region_segments = self._init_regions() if use_regions else None
        self._sampling_region_cache: dict[float, BaseGeometry] = {}
        self._sampling_region_prepared_cache = {}

    def bounds(self) -> Tuple[float, float, float, float]:
        """Coordinates of a bounding box that contains the map.

        Returns:
            x_min: Bottom left corner x coordinate [m].
            y_min: Bottom left corner y coordinate [m].
            x_max: Top right corner x coordinate [m].
            y_max: Top right corner y coordinate [m].

        """
        return self._map_polygon.bounds

    def check_collision(
        self, segment: List[Tuple[float, float]], compute_distance: bool = False
    ) -> Tuple[Tuple[float, float], float]:
        """Determines if a segment intersects with the map.

        Args:
            segment: Sensor ray or motion trajectory in the format [(start), (end)].
            compute_distance: True to compute the distance between the robot and the intersection.

        Returns:
            intersection: Closest collision point (x, y) [m].
            distance: Distance to the obstacle [m]. inf if not computed.

        """
        intersection = []
        distance = float("inf")
        x0, y0 = segment[0]
        best_d2 = float("inf")

        map_segments = self._segments_for_origin(segment[0])
        if map_segments is None:
            # Sensor rays may be outside the map even if the robot center is within it.
            return intersection, distance

        if self._intersect is not None:
            xi = ct.c_double(0.0)
            yi = ct.c_double(0.0)

            for map_segment in map_segments:
                found = self._intersect.segment_intersect(
                    ct.byref(xi),
                    ct.byref(yi),
                    segment[0][0],
                    segment[0][1],
                    segment[1][0],
                    segment[1][1],
                    map_segment[0][0],
                    map_segment[0][1],
                    map_segment[1][0],
                    map_segment[1][1],
                )

                if not found:
                    continue

                dx = xi.value - x0
                dy = yi.value - y0
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    intersection = (xi.value, yi.value)
        else:
            from intersect import Intersect

            intersect = Intersect()

            for map_segment in map_segments:
                pt = intersect.segment_intersect(segment, map_segment)
                if pt is None:
                    continue

                dx = pt[0] - x0
                dy = pt[1] - y0
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_d2 = d2
                    intersection = pt

        if intersection and compute_distance:
            distance = math.sqrt(best_d2)

        return intersection, distance

    def raycast_distances(
        self,
        origin: Tuple[float, float],
        ray_angles: np.ndarray,
        ray_range: float,
        no_hit_distance: float,
    ) -> np.ndarray:
        """Computes distances to the closest obstacle for multiple LiDAR rays.

        Args:
            origin: Ray start point (x, y) [m].
            ray_angles: Absolute ray angles [rad].
            ray_range: Ray length [m].
            no_hit_distance: Value used when no obstacle is found.

        Returns:
            A 1D array with one distance per ray [m].

        """
        distances = np.full(len(ray_angles), no_hit_distance, dtype=float)
        map_segments = self._segments_for_origin(origin)
        if map_segments is None or len(map_segments) == 0:
            return distances

        x0, y0 = origin

        if self._intersect is not None:
            xi = ct.c_double(0.0)
            yi = ct.c_double(0.0)

            for i, angle in enumerate(ray_angles):
                x1 = x0 + ray_range * math.cos(angle)
                y1 = y0 + ray_range * math.sin(angle)
                best_d2 = float("inf")

                for segment in map_segments:
                    found = self._intersect.segment_intersect(
                        ct.byref(xi),
                        ct.byref(yi),
                        x0,
                        y0,
                        x1,
                        y1,
                        segment[0][0],
                        segment[0][1],
                        segment[1][0],
                        segment[1][1],
                    )

                    if not found:
                        continue

                    dx = xi.value - x0
                    dy = yi.value - y0
                    d2 = dx * dx + dy * dy

                    if d2 < best_d2:
                        best_d2 = d2

                if best_d2 < float("inf"):
                    distances[i] = math.sqrt(best_d2)
        else:
            from intersect import Intersect

            intersect = Intersect()
            for i, angle in enumerate(ray_angles):
                ray = [
                    (x0, y0),
                    (
                        x0 + ray_range * math.cos(angle),
                        y0 + ray_range * math.sin(angle),
                    ),
                ]
                best_d2 = float("inf")

                for segment in map_segments:
                    pt = intersect.segment_intersect(ray, segment)
                    if pt is None:
                        continue

                    dx = pt[0] - x0
                    dy = pt[1] - y0
                    d2 = dx * dx + dy * dy

                    if d2 < best_d2:
                        best_d2 = d2

                if best_d2 < float("inf"):
                    distances[i] = math.sqrt(best_d2)

        return distances

    def contains(self, point: Tuple[float, float]) -> bool:
        """Determines whether a point is within the map limits.

        Args:
            point: (x, y) coordinates to check.

        Returns:
            bool: True if the point is inside the map; False otherwise.

        """
        pt = Point(point[0], point[1])

        return self._map_polygon.contains(pt)

    def bounds_with_clearance(self, clearance: float) -> Tuple[float, float, float, float] | None:
        """Bounding box of points whose center is at least `clearance` away from walls."""
        if clearance <= 0.0:
            return self.bounds()

        region = self._sampling_region(clearance)
        if region.is_empty:
            return None

        return region.bounds

    def contains_with_clearance(self, point: Tuple[float, float], clearance: float) -> bool:
        """Checks if a point is valid as robot center given obstacle clearance."""
        if clearance <= 0.0:
            return self.contains(point)

        region = self._sampling_region(clearance)
        if region.is_empty:
            return False

        key = self._clearance_key(clearance)
        prepared = self._sampling_region_prepared_cache.get(key)
        if prepared is None:
            prepared = prep(region)
            self._sampling_region_prepared_cache[key] = prepared

        return prepared.contains(Point(point[0], point[1]))

    @property
    def grid_map(self) -> np.ndarray:
        """Grid map getter.

        Returns:
            A 2D matrix containing 1 in cells with obstacles and 0 elsewhere. None if not available.

        """
        return self._grid_map

    def plot(self, axes):
        """Draws the map.

        Args:
            axes: Figure axes.

        Returns:
            axes: Modified axes.

        """
        x_min, y_min, x_max, y_max = self.bounds()
        axis_margin = 0.1
        x_min_plot = x_min - axis_margin
        y_min_plot = y_min - axis_margin
        x_max_plot = x_max + axis_margin
        y_max_plot = y_max + axis_margin

        major_ticks = np.arange(
            math.floor(min(x_min_plot, y_min_plot)),
            math.ceil(max(x_max_plot, y_max_plot)) + 0.01,
            1,
        )
        minor_ticks = np.arange(
            math.floor(min(x_min_plot, y_min_plot)),
            math.ceil(max(x_max_plot, y_max_plot)) + 0.01,
            0.5,
        )

        axes.set_xticks(major_ticks)
        axes.set_xticks(minor_ticks, minor=True)
        axes.set_yticks(major_ticks)
        axes.set_yticks(minor_ticks, minor=True)

        axes.set_xlim(x_min_plot, x_max_plot)
        axes.set_ylim(y_min_plot, y_max_plot)
        axes.grid(which="both", alpha=0.33, linestyle="dashed", zorder=1)
        axes.set(xlabel="x [m]", ylabel="y [m]")

        # Plot map
        x, y = self._map_polygon.exterior.xy
        axes.plot(x, y, color="black", alpha=1, linewidth=3, solid_capstyle="round", zorder=2)

        for interior in self._map_polygon.interiors:
            x, y = interior.xy
            axes.plot(
                x,
                y,
                color="black",
                alpha=1,
                linewidth=3,
                solid_capstyle="round",
                zorder=2,
            )

        return axes

    def show(
        self,
        title: str,
        figure_number: int = 1,
        block: bool = True,
        figure_size: Tuple[float, float] = (7, 7),
        save_figure: bool = False,
        save_dir: str = "maps",
    ):
        """Displays the map in a figure.

        Args:
            title: Plot title.
            figure_number: Any existing figure with the same value will be overwritten.
            block: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        figure, axes = plt.subplots(1, 1, figsize=figure_size, num=figure_number)
        axes = self.plot(axes)
        axes.set_title(f"Map ({title})")
        figure.tight_layout()  # Reduce white margins

        plt.show(block=block)
        plt.pause(0.0001)  # Wait for 0.1 ms or the figure won't be displayed

        if save_figure:
            save_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", save_dir))

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = title.lower() + ".png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def show_regions(
        self,
        title: str,
        figure_number: int = 1,
        block: bool = True,
        figure_size: Tuple[float, float] = (7, 7),
        save_figure: bool = False,
        save_dir: str = "maps",
    ):
        """Displays the map segments that belong to each region.

        Args:
            title: Plot title.
            figure_number: Any existing figure with the same value will be overwritten.
            block: True to stop program execution until the figure window is closed.
            figure_size: Figure window dimensions.
            save_figure: True to save figure to a .png file.
            save_dir: Image save directory.

        """
        x_min, y_min, x_max, y_max = self.bounds()
        rows, cols = self._region_segments.shape
        major_ticks = np.arange(min(x_min, y_min), max(x_max, y_max) + 0.01, 1)

        if rows <= 5 and cols <= 5:
            label_size = 8.0
            map_line_width = 2.0
            marker_size = 5.0
            figure, axes = plt.subplots(
                rows,
                cols,
                figsize=figure_size,
                num=figure_number,
                sharex=True,
                sharey=True,
            )
        else:
            label_size = 5.0
            map_line_width = 1.75
            marker_size = 1.5
            figure, axes = plt.subplots(
                rows,
                cols,
                figsize=figure_size,
                num=figure_number,
                sharex=True,
                sharey=True,
                gridspec_kw={"hspace": 0, "wspace": 0},
            )

        for ax in axes.flat:
            ax.set_xlabel("x [m]", fontsize="small")
            ax.set_ylabel("y [m]", fontsize="small")
            ax.label_outer()  # Hide x and tick labels for top plots and y ticks for right plots.

            ax.set_xticks(major_ticks)
            ax.set_yticks(major_ticks)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.tick_params(axis="x", labelsize=label_size, rotation=90)
            ax.tick_params(axis="y", labelsize=label_size)

            ax.grid(which="both", alpha=0.33, linestyle="dashed", zorder=1)

        figure.suptitle(f"Map regions ({title})")
        figure.tight_layout()  # Reduce white margins

        for y in np.arange(y_max - 0.5, y_min, -1):
            for x in np.arange(x_min + 0.5, x_max):
                circle = Point(x, y).buffer(self._sensor_range + 1 / math.sqrt(2))
                cx, cy = circle.exterior.xy

                r, c = m._xy_to_rc((x, y))
                axes[r, c].plot(x, y, "bo", markersize=marker_size)
                axes[r, c].plot(
                    cx,
                    cy,
                    color="green",
                    alpha=1,
                    linewidth=1,
                    linestyle="dashed",
                    zorder=3,
                )

                for s in m._region_segments[r][c]:
                    lx, ly = LineString(s).xy
                    axes[r, c].plot(
                        lx,
                        ly,
                        color="black",
                        linewidth=map_line_width,
                        solid_capstyle="round",
                        zorder=2,
                    )

        plt.show(block=block)

        if save_figure:
            save_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", save_dir))

            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            file_name = title.lower() + "_regions.png"
            file_path = os.path.join(save_path, file_name)
            figure.savefig(file_path)

    def _init_intersect(self) -> ct.CDLL:
        """Loads a C library to compute intersections faster.

        Returns:
            intersect: An object to call functions in the library.
        """
        library_names = {
            "Windows": "libintersect.dll",
            "Darwin": "libintersect.dylib",
            "Linux": "libintersect.so",
        }

        library_path = os.path.join(os.path.dirname(__file__), library_names[platform.system()])
        intersect = ct.CDLL(library_path)

        # Initialize function arguments and return value types
        intersect.segment_intersect.restype = ct.c_bool
        intersect.segment_intersect.argtypes = [
            ct.POINTER(ct.c_double),  # xi
            ct.POINTER(ct.c_double),  # yi
            ct.c_double,
            ct.c_double,  # x0, y0
            ct.c_double,
            ct.c_double,  # x1, y1
            ct.c_double,
            ct.c_double,  # x2, y2
            ct.c_double,
            ct.c_double,
        ]  # x3, y3

        return intersect

    def _init_regions(self) -> np.ndarray:
        """Divides the map in 1x1 m squares and finds the potentially visible segments.

        This function can be further improved by considering occlusions.

        Returns:
            region_segments: A 2D matrix that contains the segments for each region.
        """
        # Obtain map dimensions
        x_min, y_min, x_max, y_max = self.bounds()
        map_rows, map_cols = math.ceil(y_max - y_min), math.ceil(x_max - x_min)

        # Precomputed constants to convert from (x, y) to (row, col) faster
        self._XC = math.floor(map_cols / 2.0)
        self._YR = map_rows - math.ceil(map_rows / 2.0)

        # Find the segments visible from each region
        region_segments = np.zeros((map_rows, map_cols), dtype=list)

        for y in np.arange(y_max - 0.5, y_min, -1):
            for x in np.arange(x_min + 0.5, x_max):
                circle = Point(x, y).buffer(self._sensor_range + 1 / math.sqrt(2))
                segments = []

                for segment in self._map_segments:
                    line = LineString(segment)

                    if line.intersects(circle) and not line.touches(circle):
                        segments.append(segment)

                r, c = self._xy_to_rc((x, y))
                region_segments[r][c] = segments

        return region_segments

    def _xy_to_rc(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """Converts (x, y) coordinates of a metric map to (row, col) coordinates of a grid map.

        Args:
            xy: (x, y) [m].

        Returns:
            rc: (row, col) starting from (0, 0) at the top left corner.

        """
        x = math.floor(xy[0])
        y = math.ceil(xy[1])

        row = max(0, int(self._YR - y))
        col = max(0, int(x + self._XC))

        return row, col

    def _segments_for_origin(self, origin: Tuple[float, float]):
        """Returns candidate map segments for a given origin point."""
        if self._region_segments is None:
            return self._map_segments

        r, c = self._xy_to_rc(origin)
        rows, cols = self._region_segments.shape
        if r < 0 or c < 0 or r >= rows or c >= cols:
            return None

        return self._region_segments[r][c]

    @staticmethod
    def _clearance_key(clearance: float) -> float:
        """Rounds clearance to keep cache keys stable under float noise."""
        return round(float(clearance), 6)

    def _sampling_region(self, clearance: float) -> BaseGeometry:
        """Returns free-space region where robot center can be placed."""
        key = self._clearance_key(clearance)
        region = self._sampling_region_cache.get(key)
        if region is not None:
            return region

        region = self._map_polygon.buffer(-float(clearance))
        self._sampling_region_cache[key] = region
        return region


if __name__ == "__main__":
    # Display the full map and its regions
    map_name = "lab02"
    map_path = os.path.realpath(
        os.path.join(os.path.dirname(__file__), "..", "maps", map_name + ".json")
    )

    m = Map(map_path, sensor_range=1.0)
    m.show(
        title=map_name,
        figure_number=1,
        block=False,
        figure_size=(8, 8),
        save_figure=True,
    )
    m.show_regions(title=map_name, figure_number=2, figure_size=(8, 8), save_figure=True)
