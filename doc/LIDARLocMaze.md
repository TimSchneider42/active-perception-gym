# LIDARLocMaze

<p align="center"><img src="img/LIDARLocMaze-v0.gif" alt="LIDARLocMaze-v0" width="200px"/></p>

This environment is part of the LIDAR localization environments.
Refer to the [LIDAR Localization environments overview](LIDARLocalization.md) for a general description of these environments.

## Environment Details

|                    |                 |
|--------------------|-----------------|
| **Environment ID** | LIDARLocMaze-v0 |
| **Map size**       | 21x21           |
| **Map type**       | Maze            |
| **Static/dynamic** | dynamic         |

## Description

In the LIDARLocMaze environment, the agent faces a map with narrow corridors.
Hence, it will always receive information from its LIDAR sensors, but many regions of the maze look alike.
The agent must navigate around the map to gather information and localize itself.
In this variant, the maze layout changes every episode, meaning that the agent has to learn to process the map it is provided as additional input.

## Version History

- `v0`: Initial release.
