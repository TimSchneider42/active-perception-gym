# LIDARLocMazeStatic

<p align="center"><img src="img/LIDARLocRoomsStatic-v0.gif" alt="LIDARLocRoomsStatic-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                           |               |
|---------------------------|-----------------|
| **Environment ID**        | LIDARLocMazeStatic-v0 |
| **Map type**              | Grayscale       |
| **Map size**              | 21x21           |
| **LIDAR beams**           | 8               |
| **LIDAR range**           | 5.0             |

## Description

In the LIDARLocRoomsStatic environment, the agent's objective is to localize itself in a room.
The room floorplan remains unchanged across all episodes. Here, the agent has limited perception, relying only on:
- *Odometry*, which provides relative movement information.
- *LIDAR readings* from 8 beams with a maximum range of 5.0 units.

Since the entire environment is not visible at once, the agent must move strategically to gather enough information for accurate localization.


## Version History

- `v0`: Initial release.
