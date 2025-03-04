import io
import math
import random
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image

from ap_gym import ActivePerceptionEnv, ActivePerceptionActionSpace, MSELossFn


class LIDARLocalization2DEnv(
    ActivePerceptionEnv[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Literal["rgb_array"] = "rgb_array"):
        super().__init__()
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")

        self.__maze_width = 28
        self.__maze_height = 28
        self.__scale = 30
        self.__num_rooms = 5
        self.__branching_prob = 1.0

        self.__lidar_range = 100
        self.__lidar_directions = [i * (math.pi / 4) for i in range(8)]  # 8 directions

        maze = self._generate_maze(
            self.__maze_width,
            self.__maze_height,
            self.__num_rooms,
            self.__branching_prob,
        )
        maze = np.array(maze, dtype=np.uint8) * 255
        self.__base_image = maze

        self.__bitmap = self._maze_to_pygame_surface(
            self.__base_image, scale=self.__scale
        )
        self.__screen_size = (600, 600)
        self.__bitmap = pygame.transform.scale(
            self.__bitmap,
            self.__screen_size,
        )
        pygame.init()

        self.__screen = pygame.display.set_mode(self.__screen_size)
        self.__clock = pygame.time.Clock()

        coords_x, coords_y = np.meshgrid(
            np.linspace(-1, 1, self.__maze_width),
            np.linspace(-1, 1, self.__maze_height),
            indexing="ij",
        )

        self.action_space = ActivePerceptionActionSpace(
            gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
        )
        self.prediction_target_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-2, high=2, shape=(2,), dtype=np.float32
        )
        self.loss_fn = MSELossFn()

    def __get_obs(self):
        distances = self.__lidar_scan(self._x, self._y)
        self.__last_obs = distances
        return self.__last_obs

    def _reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        self.__rng = np.random.default_rng(seed)

        while True:
            self.__pos = self.__rng.uniform(
                np.array([0.0, -1.0]), np.ones(2), size=2
            ).astype(np.float32)

            x_idx = int(
                (self.__pos[0] + 1) / 2 * (self.__screen_size[0] - 1)
            )  # Map 0-1 to maze grid x
            y_idx = int(
                (self.__pos[1] + 1) / 2 * (self.__screen_size[1] - 1)
            )  # Map -1 to 1 to maze grid y
            pixel_array = pygame.surfarray.array3d(self.__bitmap)
            print(pixel_array[x_idx, y_idx, 0])
            print(x_idx, y_idx)
            if pixel_array[x_idx, y_idx, 0] == 255:
                self.__pos = self.__pos
                self._x = x_idx
                self._y = y_idx
                break  # Exit loop once a valid position is found
        return self.__get_obs(), {}, self.__pos

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        self.__last_pred = prediction
        base_reward = np.sum(action**2, axis=-1)
        action_clipped = np.clip(action, -1, 1)
        self.__pos += action_clipped * 0.05
        terminated = False
        if np.any(np.abs(self.__pos) >= 1):
            base_reward -= 20
            terminated = True
        self.__pos = np.clip(self.__pos, -1, 1)
        return self.__get_obs(), base_reward, terminated, False, {}, self.__pos

    def render(self):
        new_x = int(
            (self.__pos[0] + 1) / 2 * (self.__screen_size[0] - 1)
        )  # Map 0-1 to maze grid x
        new_y = int(
            (self.__pos[1] + 1) / 2 * (self.__screen_size[1] - 1)
        )  # Map -1 to 1 to maze grid y
        self.__screen.blit(self.__bitmap, (0, 0))
        if self._can_move(new_x, new_y):
            self._x, self._y = new_x, new_y

        for angle, dist in zip(self.__lidar_directions, self.__last_obs):
            end_x = int(self._x + math.cos(angle) * dist)
            end_y = int(self._y + math.sin(angle) * dist)
            pygame.draw.line(
                self.__screen, (0, 255, 0), (self._x, self._y), (end_x, end_y), 1
            )
            pygame.draw.circle(self.__screen, (255, 0, 0), (end_x, end_y), 3)

        pygame.draw.circle(self.__screen, (0, 0, 255), (self._x, self._y), 5)
        pygame.display.flip()
        self.__clock.tick(6)

        # pygame.draw.circle(screen, (0, 0, 255), (x, y), 5)

        return self.__bitmap

    def _generate_maze(self, width, height, num_rooms, branching_prob=1.0):
        # Ensure dimensions are odd
        if width % 2 == 0:
            width += 1
        if height % 2 == 0:
            height += 1

        # Create a maze full of walls (represented by 1)
        maze = [[1 for _ in range(width)] for _ in range(height)]

        def carve(x, y):
            directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
            random.shuffle(directions)
            first = True  # Always carve at least one direction to ensure connectivity
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < width and 0 < ny < height and maze[ny][nx] == 1:
                    # Always carve the first eligible branch; subsequent ones only if allowed by branching_prob
                    if first or random.random() < branching_prob:
                        maze[y + dy // 2][
                            x + dx // 2
                        ] = 0  # Carve passage between cells
                        maze[ny][nx] = 0
                        carve(nx, ny)
                        first = False

        # Start carving from the initial cell
        maze[1][1] = 0
        carve(1, 1)

        # Add random open areas (rooms)
        for _ in range(num_rooms):
            area_x = random.randint(2, width - 8)
            area_y = random.randint(2, height - 8)
            area_width = random.randint(3, 7)
            area_height = random.randint(3, 7)
            for i in range(area_y, min(area_y + area_height, height - 1)):
                for j in range(area_x, min(area_x + area_width, width - 1)):
                    maze[i][j] = 0  # Open space (white)

        return maze

    def _maze_to_pygame_surface(self, maze, scale=10):
        height = len(maze)
        width = len(maze[0])
        img = Image.new("1", (width * scale, height * scale), 1)  # white background
        for y in range(height):
            for x in range(width):
                color = (
                    1 if maze[y][x] == 0 else 0
                )  # 1: white for path, 0: black for wall
                for i in range(scale):
                    for j in range(scale):
                        img.putpixel((x * scale + j, y * scale + i), color)
        # Convert the PIL image to RGB mode (for Pygame compatibility)
        img = img.convert("RGB")
        # Save image to a bytes buffer and load directly into Pygame
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        pygame_img = pygame.image.load(img_bytes)
        return pygame_img

    # Check if pixel at position is white (traversable)
    def _can_move(self, pos_x, pos_y):
        if (
            0 <= pos_x < self.__screen_size[0]
            and 0 <= pos_y < self.__screen_size[1]
        ):
            return self.__bitmap.get_at((int(pos_x), int(pos_y)))[:3] == (255, 255, 255)
        return False

    # LiDAR sensor function
    def __lidar_scan(self, x, y):
        distances = []
        for angle in self.__lidar_directions:
            dist = 0
            while dist < self.__lidar_range:
                check_x = int(x + math.cos(angle) * dist)
                check_y = int(y + math.sin(angle) * dist)
                if not self._can_move(check_x, check_y):
                    break
                dist += 1
            distances.append(dist)
        return distances
