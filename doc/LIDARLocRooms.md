# LIDARLocMaze

<p align="center"><img src="img/LIDARLocRooms-v0.gif" alt="LIDARLocRooms-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.  
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                           |               |
|---------------------------|-----------------|
| **Environment ID**        | LIDARLocMaze-v0 |
| **Map type**              | Grayscale       |
| **Map size**              | 32x32           |
| **LIDAR beams**           | 8               |
| **LIDAR range**           | 5.0             |

## Description

In the LIDARLocRooms environment, the agent's objective is to localize itself inside a room.  
Unlike static environments, the room's floorplan changes each episode. To account for this variability, the agent is provided with a map of the current room as input. Here, the agent has limited perception, relying only on:
- *Odometry*, which provides relative movement information.
- *LIDAR readings* from 8 beams with a maximum range of 5.0 units.
- *Map*, a full grayscale representation of the current environment to assist in localization.

## Version History

- `v0`: Initial release.
