# LIDARLocMazeStatic

<p align="center"><img src="img/LIDARLocRoomsStatic-v0.gif" alt="LIDARLocRoomsStatic-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                    |                        |
|--------------------|------------------------|
| **Environment ID** | LIDARLocRoomsStatic-v0 |
| **Map size**       | 32x32                  |
| **Map type**       | Rooms                  |
| **Static/dynamic** | static                 |

## Description

In the LIDARLocRooms environment, the agent faces a map with wide open areas.
Hence, often it might not receive any information from its LIDAR sensors if it is in the middle of a large room.
The agent must, thus, navigate around the map to gather information and localize itself.
In this variant, the map stays constant, meaning that the agent can memorize the layout of the rooms over the course of the training.


## Version History

- `v0`: Initial release.
