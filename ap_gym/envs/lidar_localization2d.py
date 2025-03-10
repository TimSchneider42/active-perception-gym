from abc import abstractmethod
from typing import Any, Literal

import PIL.Image
import PIL.ImageDraw
import gymnasium as gym
import numpy as np
import shapely

from ap_gym import ActiveRegressionEnv, ImageSpace


class LIDARLocalization2DEnv(ActiveRegressionEnv[np.ndarray, np.ndarray]):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        map_width: int,
        map_height: int,
        render_mode: Literal["rgb_array"] = "rgb_array",
        static_map: bool = True,
        lidar_beam_count: int = 8,
        lidar_range: float = 5,
    ):
        super().__init__(
            2, gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        )
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")

        self.__static_map = static_map
        self.__map_width = map_width
        self.__map_height = map_height

        self.__lidar_range = lidar_range
        lidar_angles = np.linspace(
            -np.pi, np.pi, lidar_beam_count, dtype=np.float32, endpoint=False
        )
        self.__lidar_directions = (
            np.stack([np.cos(lidar_angles), np.sin(lidar_angles)], axis=-1)
            * self.__lidar_range
        )

        observation_dict = {
            "lidar": gym.spaces.Box(
                low=0, high=1, shape=(lidar_beam_count,), dtype=np.float32
            ),
            "odometry": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        }

        if not static_map:
            observation_dict["map"] = ImageSpace(
                width=self.__map_width, height=self.__map_height, channels=1
            )

        self.observation_space = gym.spaces.Dict(observation_dict)

        self.__map = None
        self.__map_shapely = None
        self.__map_obs = None
        self.__pos = None
        self.__last_pred = None
        self.__initial_pos = None

    def __get_obs(self):
        self.__last_lidar_readings = distances = self.__lidar_scan(
            self.__pos, self.__pos + self.__lidar_directions
        )
        odometry = self.__pos - self.__initial_pos
        odometry_max_value = np.array(
            [self.__map_width, self.__map_height], dtype=np.float32
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

    @abstractmethod
    def _get_map(self, seed: int):
        pass

    def _reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        self.__last_lidar_readings = None
        self.__rng = np.random.default_rng(seed)

        if not self.__static_map or self.__map is None:
            map_seed = (
                0
                if self.__static_map
                else self.__rng.integers(0, 2**32 - 1, endpoint=True)
            )
            new_map = self._get_map(map_seed)
            assert new_map.shape == (self.__map_height, self.__map_width)
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
        if not self.__static_map:
            self.__map_obs = self.__map[..., None].astype(np.float32) / 255
        valid_starting_coords = np.where(self.__map == 0)
        idx = self.__rng.integers(0, len(valid_starting_coords[0]))
        self.__pos = self.__initial_pos = (
            np.array(
                [valid_starting_coords[1][idx], valid_starting_coords[0][idx]],
                dtype=np.float32,
            )
            + 0.5
        )

        return self.__get_obs(), {}, self.__pos

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
        dist_to_wall = self.__lidar_scan(self.__pos, target_pos[None])[0]
        self.__pos += direction * dist_to_wall

        # Make agent slide along wall
        remaining_dist = total_dist - dist_to_wall
        if remaining_dist > 0:
            remaining_vec = direction * remaining_dist
            direction_candidates = np.eye(2, dtype=np.float32) * remaining_vec
            target_pos_candidates = self.__pos + direction_candidates
            dist_to_wall_candidates = self.__lidar_scan(
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

        return self.__get_obs(), base_reward, terminated, False, {}, normalized_pos

    def render(self):
        width = 500
        scale = width / self.__map.shape[1]
        base_img = (
            PIL.Image.fromarray(~self.__map)
            .resize(
                (
                    int(round(self.__map.shape[1] * scale)),
                    int(round(self.__map.shape[0] * scale)),
                )
            )
            .convert("RGB")
        )

        draw = PIL.ImageDraw.Draw(base_img, mode="RGBA")
        contact_marker_radius = 0.1
        agent_radius = 0.2
        agent_color = (0, 55, 255)
        pred_color = (255, 0, 255)
        lidar_color = (55, 255, 55)
        lidar_contact_color = (255, 55, 55)

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
                    fill=lidar_color,
                )
                contact_point = self.__pos + direction * dist
                draw.ellipse(
                    (
                        (contact_point[0] - contact_marker_radius) * scale,
                        (contact_point[1] - contact_marker_radius) * scale,
                        (contact_point[0] + contact_marker_radius) * scale,
                        (contact_point[1] + contact_marker_radius) * scale,
                    ),
                    fill=lidar_contact_color,
                )
        if self.__last_pred is not None:
            draw.line(
                (
                    self.__pos[0] * scale,
                    self.__pos[1] * scale,
                    self.__last_pred[0] * scale,
                    self.__last_pred[1] * scale,
                ),
                fill=pred_color + (80,),
            )

            draw.ellipse(
                (
                    (self.__last_pred[0] - agent_radius) * scale,
                    (self.__last_pred[1] - agent_radius) * scale,
                    (self.__last_pred[0] + agent_radius) * scale,
                    (self.__last_pred[1] + agent_radius) * scale,
                ),
                fill=pred_color,
            )

        draw.ellipse(
            (
                (self.__pos[0] - agent_radius) * scale,
                (self.__pos[1] - agent_radius) * scale,
                (self.__pos[0] + agent_radius) * scale,
                (self.__pos[1] + agent_radius) * scale,
            ),
            fill=agent_color,
        )

        return np.array(base_img)

    # LiDAR sensor function
    def __lidar_scan(self, pos: np.ndarray, target_pos: np.ndarray, eps=1e-3):
        output = np.empty(target_pos.shape[0], dtype=np.float32)
        for i, target in enumerate(target_pos):
            line = shapely.LineString([pos, target])
            intersections = line.intersection(self.__map_shapely)
            if (
                isinstance(intersections, shapely.LineString)
                and not intersections.is_empty
            ):
                output[i] = (
                    np.linalg.norm(
                        np.array([intersections.xy[0][0], intersections.xy[1][0]]) - pos
                    )
                    - eps
                )
            elif isinstance(intersections, shapely.Point):
                output[i] = 0
            elif isinstance(intersections, shapely.MultiPoint) or isinstance(
                intersections, shapely.MultiLineString
            ):
                intersection_points = np.array(
                    [[p.xy[0][0], p.xy[1][0]] for p in intersections.geoms],
                    dtype=np.float32,
                )
                distances = np.linalg.norm(intersection_points - pos, axis=-1)
                output[i] = np.min(distances) - eps
            else:
                output[i] = np.linalg.norm(target - pos)
        return output
