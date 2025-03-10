from collections import deque
from typing import Any, Literal

import PIL.Image
import PIL.ImageDraw
import gymnasium as gym
import numpy as np
import shapely

from ap_gym import ActiveRegressionEnv, ImageSpace
from .dataset import DataLoader, DatasetIterator
from .floor_map import FloorMapDataset
from .style import (
    COLOR_AGENT,
    COLOR_PRED,
    COLOR_OBS_PRIMARY,
    COLOR_OBS_SECONDARY,
    quality_color,
)


class LIDARLocalization2DEnv(ActiveRegressionEnv[np.ndarray, np.ndarray]):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        dataset: FloorMapDataset,
        render_mode: Literal["rgb_array"] = "rgb_array",
        static_map: bool = False,
        lidar_beam_count: int = 8,
        lidar_range: float = 5,
        static_map_index: int = 0,
        prefetch: bool = True,
        prefetch_buffer_size: int = 128,
    ):
        super().__init__(
            2, gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")

        self.__static_map = static_map
        self.__dataset = dataset
        self.__prefetch = prefetch
        self.__prefetch_buffer_size = prefetch_buffer_size

        self.__map = None
        self.__map_shapely = None
        self.__map_obs = None
        self.__pos = None
        self.__last_pred = None
        self.__initial_pos = None
        self.__trajectory = deque()
        self.__observation_map = None
        self.__data_loader = None
        self.__np_random = None
        self.__map_idx = None

        self.__dataset.load()
        if self.__static_map:
            self.__set_map(self.__dataset[static_map_index], static_map_index)

        self.__lidar_range = lidar_range
        lidar_angles = np.linspace(
            -np.pi, np.pi, lidar_beam_count, dtype=np.float32, endpoint=False
        )
        lidar_directions_unscaled = np.stack(
            [np.cos(lidar_angles), np.sin(lidar_angles)], axis=-1
        )
        self.__lidar_directions = lidar_directions_unscaled * self.__lidar_range
        self.__scan_points = (
            np.arange(0, self.__lidar_range, 0.05)[None, :, None]
            * lidar_directions_unscaled[:, None]
        )

        observation_dict = {
            "lidar": gym.spaces.Box(
                low=0, high=1, shape=(lidar_beam_count,), dtype=np.float32
            ),
            "odometry": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }

        if not static_map:
            observation_dict["map"] = ImageSpace(
                width=dataset.map_width, height=dataset.map_height, channels=1
            )

        self.observation_space = gym.spaces.Dict(observation_dict)

    def __get_obs(self):
        distances, contact_coords = self.__lidar_scan(
            self.__pos, self.__pos + self.__lidar_directions
        )
        self.__last_lidar_readings = distances

        valid_contact_coords = contact_coords[np.all(contact_coords >= 0, axis=-1)]
        self.__observation_map[
            valid_contact_coords[:, 1], valid_contact_coords[:, 0]
        ] = True

        # Not 100% exact but good enough for visualization
        valid = np.linalg.norm(self.__scan_points, axis=-1) <= distances[..., None]
        observed_pos = np.floor(self.__pos[None, None] + self.__scan_points).astype(
            np.int_
        )
        observed_pos = observed_pos[valid]
        self.__observation_map[observed_pos[..., 1], observed_pos[..., 0]] = True

        odometry = self.__pos - self.__initial_pos
        odometry_max_value = np.array(
            [self.__dataset.map_width, self.__dataset.map_height], dtype=np.float32
        )
        odometry_min_value = -odometry_max_value
        odometry_norm = (odometry - odometry_min_value) / (
            odometry_max_value - odometry_min_value
        ) * 2 - 1
        obs = {
            "lidar": distances / self.__lidar_range,
            "odometry": odometry_norm,
        }
        if not self.__static_map:
            obs["map"] = self.__map_obs
        return obs

    def __set_map(self, new_map: np.ndarray, map_idx: int):
        assert new_map.shape == (self.__dataset.map_height, self.__dataset.map_width)
        self.__map = new_map
        coords = np.meshgrid(
            np.arange(self.__map.shape[1]), np.arange(self.__map.shape[0])
        )
        occupied_coords = np.stack(
            [coords[0][self.__map], coords[1][self.__map]], axis=-1
        )
        self.__map_shapely = shapely.union_all(
            [shapely.box(*c, *(c + 1)) for c in occupied_coords]
        )
        self.__map_idx = map_idx

    def _reset(self, *, options: dict[str, Any | None] = None):
        self.__last_lidar_readings = None

        if not self.__static_map:
            self.__set_map(*next(self.__data_loader))
            self.__map_obs = self.__map[..., None].astype(np.float32) / 255

        self.__observation_map = np.zeros_like(self.__map, dtype=np.bool_)

        valid_starting_coords = np.where(self.__map == 0)
        idx = self.np_random.integers(0, len(valid_starting_coords[0]))
        self.__pos = self.__initial_pos = (
            np.array(
                [valid_starting_coords[1][idx], valid_starting_coords[0][idx]],
                dtype=np.float32,
            )
            + 0.5
        )
        self.__trajectory.clear()
        self.__trajectory.append((self.__pos, None))

        return self.__get_obs(), {"map_idx": self.__map_idx}, self.__pos

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        map_size = np.array(
            [self.__map.shape[1], self.__map.shape[0]], dtype=np.float32
        )
        self.__last_pred = (prediction + 1) / 2 * map_size

        base_reward = np.sum(action**2, axis=-1)
        action_clipped = np.clip(action, -1, 1)
        target_pos = self.__pos + action_clipped
        direction = target_pos - self.__pos
        total_dist = np.linalg.norm(direction)
        direction /= total_dist
        dist_to_wall, _ = self.__lidar_scan(self.__pos, target_pos[None])
        dist_to_wall = dist_to_wall[0]
        self.__pos += direction * dist_to_wall

        # Make agent slide along wall
        remaining_dist = total_dist - dist_to_wall
        if remaining_dist > 0:
            remaining_vec = direction * remaining_dist
            direction_candidates = np.eye(2, dtype=np.float32) * remaining_vec
            target_pos_candidates = self.__pos + direction_candidates
            dist_to_wall_candidates, _ = self.__lidar_scan(
                self.__pos, target_pos_candidates
            )
            if dist_to_wall_candidates[0] > 0:
                idx = 0
            else:
                idx = 1
            self.__pos += (
                direction_candidates[idx]
                / np.linalg.norm(direction_candidates[idx])
                * dist_to_wall_candidates[idx]
            )

        terminated = False
        pos_min = np.zeros(2, dtype=np.float32)
        pos_max = np.array([self.__map.shape[1], self.__map.shape[0]], dtype=np.float32)
        if np.any(self.__pos < pos_min) or np.any(self.__pos >= pos_max):
            base_reward -= 20
            terminated = True
        self.__pos = np.clip(
            self.__pos,
            pos_min,
            pos_max,
        )

        normalized_pos = self.__pos / map_size * 2 - 1

        prediction_quality = np.linalg.norm(
            prediction - normalized_pos, axis=-1
        ) / np.sqrt(4)
        self.__trajectory.append((self.__pos, np.clip(prediction_quality, 0, 1)))

        return (
            self.__get_obs(),
            base_reward,
            terminated,
            False,
            {"map_idx": self.__map_idx},
            normalized_pos,
        )

    def render(self):
        width = 500
        scale = width / self.__map.shape[1]
        alpha = 0.25 + 0.75 * self.__observation_map.astype(np.float32)
        base_img = (
            PIL.Image.fromarray(
                (alpha * (~self.__map).astype(np.float32) + (1.0 - alpha) * 0.5) * 255
            )
            .resize(
                (
                    int(round(self.__map.shape[1] * scale)),
                    int(round(self.__map.shape[0] * scale)),
                ),
                resample=PIL.Image.NEAREST,
            )
            .convert("RGB")
        )

        draw = PIL.ImageDraw.Draw(base_img, mode="RGBA")
        contact_marker_radius = 0.2
        agent_radius = 0.2

        traj_hist = list(self.__trajectory)
        for (pos_a, qual_a), (pos_b, qual_b) in zip(
            traj_hist[:-1], list(traj_hist)[1:]
        ):
            draw.line(
                (
                    pos_a[0] * scale,
                    pos_a[1] * scale,
                    pos_b[0] * scale,
                    pos_b[1] * scale,
                ),
                width=2,
                fill=quality_color(qual_b),
            )

        if self.__last_lidar_readings is not None:
            directions_norm = self.__lidar_directions / np.linalg.norm(
                self.__lidar_directions, axis=-1, keepdims=True
            )
            for dist, direction in zip(self.__last_lidar_readings, directions_norm):
                draw.line(
                    (
                        self.__pos[0] * scale,
                        self.__pos[1] * scale,
                        (self.__pos + direction * dist)[0] * scale,
                        (self.__pos + direction * dist)[1] * scale,
                    ),
                    width=2,
                    fill=COLOR_OBS_PRIMARY,
                )
                contact_point = self.__pos + direction * dist
                draw.ellipse(
                    (
                        (contact_point[0] - contact_marker_radius) * scale,
                        (contact_point[1] - contact_marker_radius) * scale,
                        (contact_point[0] + contact_marker_radius) * scale,
                        (contact_point[1] + contact_marker_radius) * scale,
                    ),
                    fill=COLOR_OBS_SECONDARY,
                )
        if self.__last_pred is not None:
            draw.line(
                (
                    self.__pos[0] * scale,
                    self.__pos[1] * scale,
                    self.__last_pred[0] * scale,
                    self.__last_pred[1] * scale,
                ),
                fill=COLOR_PRED + (80,),
            )

            draw.ellipse(
                (
                    (self.__last_pred[0] - agent_radius) * scale,
                    (self.__last_pred[1] - agent_radius) * scale,
                    (self.__last_pred[0] + agent_radius) * scale,
                    (self.__last_pred[1] + agent_radius) * scale,
                ),
                fill=COLOR_PRED,
            )

        draw.ellipse(
            (
                (self.__pos[0] - agent_radius) * scale,
                (self.__pos[1] - agent_radius) * scale,
                (self.__pos[0] + agent_radius) * scale,
                (self.__pos[1] + agent_radius) * scale,
            ),
            fill=COLOR_AGENT,
        )

        return np.array(base_img)

    def __lidar_scan(self, pos: np.ndarray, target_pos: np.ndarray, eps=1e-3):
        output_distances = np.empty(target_pos.shape[0], dtype=np.float32)
        output_contact_coords = np.empty((target_pos.shape[0], 2), dtype=np.int32)
        for i, target in enumerate(target_pos):
            line = shapely.LineString([pos, target])
            intersections = line.intersection(self.__map_shapely)
            if (
                isinstance(intersections, shapely.LineString)
                and not intersections.is_empty
            ):
                contact_point = np.array(
                    [intersections.xy[0][0], intersections.xy[1][0]]
                )
                output_distances[i] = np.maximum(
                    np.linalg.norm(contact_point - pos) - eps, 0
                )
            elif isinstance(intersections, shapely.Point):
                contact_point = pos
                output_distances[i] = 0
            elif isinstance(intersections, shapely.MultiPoint) or isinstance(
                intersections, shapely.MultiLineString
            ):
                intersection_points = np.array(
                    [[p.xy[0][0], p.xy[1][0]] for p in intersections.geoms],
                    dtype=np.float32,
                )
                distances = np.linalg.norm(intersection_points - pos, axis=-1)
                min_idx = np.argmin(distances)
                contact_point = intersection_points[min_idx]
                output_distances[i] = np.maximum(distances[min_idx] - eps, 0)
            else:
                output_distances[i] = np.linalg.norm(target - pos)
                contact_point = target
            if output_distances[i] < np.linalg.norm(target - pos):
                coords = np.floor(contact_point)
                exact = np.abs(coords - contact_point) < 1e-5
                coords[exact & (target < pos)] -= 1
                output_contact_coords[i] = coords
            else:
                output_contact_coords[i] = -1
        return output_distances, output_contact_coords

    def close(self):
        if self.__data_loader is not None:
            self.__data_loader.close()
        super().close()

    @property
    def _np_random(self):
        return self.__np_random

    @_np_random.setter
    def _np_random(self, np_random):
        if not self.__static_map:
            self.__data_loader = DataLoader(
                DatasetIterator(
                    self.__dataset, np_random.integers(0, 2**32, endpoint=True)
                ),
                self.__prefetch,
                self.__prefetch_buffer_size,
            )
        self.__np_random = np_random
