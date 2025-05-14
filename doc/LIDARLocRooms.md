# LIDARLocMaze

<p align="center"><img src="img/LIDARLocRooms-v0.gif" alt="LIDARLocRooms-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                    |                  |
|--------------------|------------------|
| **Environment ID** | LIDARLocRooms-v0 |
| **Map size**       | 32x32            |
| **Map type**       | Rooms            |
| **Static/dynamic** | dynamic          |

## Description

In the LIDARLocRooms environment, the agent faces a map with wide open areas.
Hence, often it might not receive any information from its LIDAR sensors if it is in the middle of a large room.
The agent must, thus, navigate around the map to gather information and localize itself.
In this variant, the room layout changes every episode, meaning that the agent has to learn to process the map it is provided as additional input.

## Version History

- `v0`: Initial release.
