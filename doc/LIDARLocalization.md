# 2D LIDAR Localization Environments

In the 2D LIDAR localization environments, the agent must navigate the world and use LIDAR sensors to perceive its surroundings. From this sensory data, it is tasked with estimating its own position within the environment. The number of LIDAR beams and their ranges can vary, but they should only give the agent a partial glimpse of its surroundings.

In these environments, there are two main variants:

- **Static Environments**: In environments such as `LIDARLocMazeStatic-v0` and `LIDARLocRoomsStatic-v0`, the maze or room layout is fixed.
- **Map-based Environments**: In `LIDARLocMaze-v0` and `LIDARLocRooms-v0`, the agent is additionally provided with a map of the environment.

In both cases, the primary challenge is for the agent to accurately localize itself.

Consider the following example from the [LIDARLocRoomsStatic](LIDARLocMazeStatic.md) environment:

<p align="center"><img src="img/LIDARLocRoomsStatic-v0.gif" alt="2D LIDAR Localization" width="200px"/></p>

In this visualization, the agent moves through the room while using its LIDAR sensors, represented by the green beams extending outward. The white regions indicate areas that the agent has observed so far. To illustrate the agent’s localization confidence, we track its past positions with a color gradient ranging from red to green. A red trail means that the agent's predicted position had a probability close to 0 of being correct, whereas a green trail indicates high confidence, with a probability close to 1.

All 2D LIDAR localization environments in ap_gym are implemented as instances of the `ap_gym.envs.lidar_localization2d.LIDARLocalization2DEnv` class and share the following properties:

## Properties

| Property                          | Value                                                                                                                                                                                                                                                                                           |
|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Action Space**                  | `Box(-1.0, 1.0, shape=(2,), dtype=np.float32)`                                                                                                                                                                                                                                                  |
| **Localization Prediction Space** | `Box(-inf, inf, shape=(2,), dtype=np.float32)`                                                                                                                                                                                                                                                  |
| **Prediction Target Space**       | `Box(-inf, inf, shape=(2,), dtype=np.float32)`                                                                                                                                                                                                                                                  |
| **Observation Space**             | `Dict({`<br>`"lidar": Box(0.0, 1.0, shape=(B,), dtype=np.float32),`<br>`"odometry": Box(-1.0, 1.0, shape=(2,), dtype=np.float32),`<br>`["map": ImageSpace(width=M, height=M, channels=1, dtype=np.float32)]`<br>`})` <br><em>(The `"map"` key is only included in the static enviroments.)</em> |
| **Loss Function**                 | `ap_gym.MSELossFn()`                                                                                                                                                                                                                                                                            |

where $B \in \mathbb{N}$ is the number of LIDAR beams, and $M\in \mathbb{N} is the dimension of the provided map when available.

## Action Space

The action is an `np.ndarray` with shape `(2,)` consisting of continuous values in the interval $[-1, 1]$.

- `action[0]`: Horizontal sensor movement
- `action[1]`: Vertical sensor movement

## Prediction Space

The prediction is a 2-dimensional `np.ndarray` representing the agent's predicted coordinates.

## Prediction Target Space

The prediction target is a 2-dimensional `np.ndarray` containing the ground-truth coordinates.

## Observation Space

The observation is a dictionary with keys `"lidar"`, `"odometry"`, and `"map"`.

The `"lidar"` is a normalized array (`np.ndarray`) of shape `(B,)` with distances measured by the LIDAR sensor, each in $[-1, 1]$.

The `"odometry"` is a normalized `np.ndarray` with shape `(2,)` representing the agent's relative displacement from its starting position.

The `"map"`(provided only in non-static map environments) is a $M \times M$ `np.ndarray` providing a grayscale image representation of the environment.

representing the target glimpse of the image where each
pixel is in the range $[-1, 1]$.

## Rewards

The reward at each timestep is a sum of:

- An action penalty equal to the squared magnitude of the action: $\lVert action\rVert^2$.
- A boundary penalty of $-20$, applied only if the agent attempts to move outside the valid environment area, which also terminates the episode.

## Starting State

The agent begins at a uniformly random, valid location within the environment.

## Episode End

Episodes end either upon exceeding a maximum number of timesteps (`max_episode_steps`, default: `16`) or if the agent attempts to leave the environment's valid area.

## Arguments

| Parameter              | Type                                               | Default     | Description                                                                                               |
|------------------------|----------------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------|
| `dataset`              | `ap_gym.envs.lidar_localization2d.FloorMapDataset` |             | Dataset used for the environment containing maps.                                                         |
| `render_mode`          | `Literal["rgb_array"]`                             | "rgb_array" | Rendering mode. Currently, only "rgb_array" is supported.                                                 |
| `static_map`           | `bool`                                             | `False`     | Whether the environment uses a single fixed map (`True`) or samples different maps per episode (`False`). |
| `lidar_beam_count`     | `int`                                              | 8           | Number of beams emitted by the LIDAR sensor.                                                              |
| `lidar_range`          | `float`                                            | 5.0         | Maximum range (distance) for LIDAR sensor readings.                                                       |
| `static_map_index`     | `int`                                              | 0           | Index of the map used when `static_map=True`. Ignored otherwise.                                          |
| `prefetch`             | `bool`                                             | `True`      | Whether maps should be prefetched asynchronously for efficiency.                                          |
| `prefetch_buffer_size` | `int`                                              | 128         | Buffer size for prefetching maps from the dataset.                                                        |
| `max_episode_steps`    | `int`                                              | 16          | Maximum steps per episode.                                                                                |

## Overview of Implemented Environments

| Environment ID                                   | Map Provided | Environment Type | LIDAR beams | Map size | Description                                                                     |
|--------------------------------------------------|--------------|------------------|-------------|----------|---------------------------------------------------------------------------------|
| [LIDARLocMazeStatic-v0](LIDARLocMazeStatic.md)   | No           | Maze (static)    | 8           | 21x21    | Static maze environment with a fixed layout.                                    |
| [LIDARLocMaze-v0](LIDARLocMaze.md)               | Yes          | Maze (dynamic)   | 8           | 21x21    | Dynamic maze environment with different maps per episode.                       |
| [LIDARLocRoomsStatic-v0](LIDARLocRoomsStatic.md) | No           | Rooms (static)   | 8           | 32x32    | Static rooms environment with a fixed layout.                                   |
| [LIDARLocRooms-v0](LIDARLocRooms.md)             | Yes          | Rooms (dynamic)  | 8           | 32x32    | Dynamic rooms environment with varying layouts, provided as input to the agent. |
