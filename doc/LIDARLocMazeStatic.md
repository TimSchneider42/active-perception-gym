# LIDARLocMazeStatic

<p align="center"><img src="img/LIDARLocMazeStatic-v0.gif" alt="LIDARLocMazeStatic-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                    |                       |
|--------------------|-----------------------|
| **Environment ID** | LIDARLocMazeStatic-v0 |
| **Map size**       | 21x21                 |
| **Map type**       | Maze                  |
| **Static/dynamic** | static                |

## Description

In the LIDARLocMazeStatic environment, the agent faces a map with narrow corridors.
Hence, it will always receive information from its LIDAR sensors, but many regions of the maze look alike.
The agent must navigate around the map to gather information and localize itself.
In this variant, the map stays constant, meaning that the agent can memorize the layout of the maze over the course of the training.

## Version History

- `v0`: Initial release.
