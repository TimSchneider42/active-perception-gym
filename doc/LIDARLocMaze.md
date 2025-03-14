# LIDARLocMaze

<p align="center"><img src="img/LIDARLocMaze-v0.gif" alt="LIDARLocMaze-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                           |               |
|---------------------------|-----------------|
| **Environment ID**        | LIDARLocMaze-v0 |
| **Map type**              | Grayscale       |
| **Map size**              | 21x21           |
| **LIDAR beams**           | 8               |
| **LIDAR range**           | 5.0             |

## Description

In the LIDARLocMaze environment, the agent's objective is to localize itself in a maze.
Unlike static environments, the maze layout changes each episode. To account for this variability, the agent is provided with a map of the current maze as input. Here, the agent has limited perception, relying only on:
- *Odometry*, which provides relative movement information.
- *LIDAR readings* from 8 beams with a maximum range of 5.0 units.
- *Map*, a full grayscale representation of the current environment to assist in localization.

## Version History

- `v0`: Initial release.
