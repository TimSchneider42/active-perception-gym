# Image Localization Environments

In image classification environments, the agent has to localize a given part of the image by moving a small glimpse around.
The glimpse is never large enough to see the entire image at once, so the agent has to move around to gather information.
Unlike the [Image Classification Environments](ImageClassification.md), this task is a regression task where the agent has to predict the coordinates of the target glimpse it is provided.

Consider the following example from the [TinyImageNetLoc](TinyImageNetLoc.md) environment:
<p align="center"><img src="img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/></p>
Marked in blue is the agent's current glimpse.
The transparent purple box represents the target glimpse the agent has to predict the coordinates of and the opaque purple box is the agent's current prediction.
We further mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that the prediction it took at this step was far from the target and green meaning that the prediction was close to the target.

All image classification environments in _ap_gym_ are instantiations of the `ap_gym.envs.image_classification.ImageLocalizationVectorEnv` class and share the following properties:

## Properties

<table>
    <tr>
        <td><strong>Action Space</strong></td>
        <td><code>Box(-1.0, 1.0, shape=(2,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Space</strong></td>
        <td><code>Box(-inf, inf, shape=(2,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Target Space</strong></td>
        <td><code>Box(-inf, inf, shape=(2,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Observation Space</strong></td>
        <td>
            <code>Dict({</code><br>
            <code>&nbsp;&nbsp;"glimpse": ImageSpace(width=G, height=G, channels=C, dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"glimpse_pos": Box(-1.0, 1.0, shape=(2,), dtype=np.float32)</code><br>
            <code>&nbsp;&nbsp;"target_glimpse": ImageSpace(width=G, height=G, channels=C, dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"time_step": Box(-1.0, 1.0, shape=(), dtype=np.float32),</code><br>
            <code>})</code>
        </td>
    </tr>
    <tr>
        <td><strong>Loss Function</strong></td>
        <td>
            <code>ap_gym.MSELossFn()</code>
        </td>
    </tr>
</table>


where $G \in \mathbb{N}$ is the glimpse size
and $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB).

## Action Space

The action is an `np.ndarray` of type `float32` with shape `(2,)`:

| Index | Description                                        |
|-------|----------------------------------------------------|
| 0     | Horizontal sensor movement in the range $[-1, 1]$. |
| 1     | Vertical sensor movement in the range $[-1, 1]$.   |

The sensor movement is scaled by the environment's `image_perception_config.max_step_length`, which is 0.2 (20% of the image) by default.

## Prediction Space

The prediction is a 2-dimensional `np.ndarray` containing the coordinates of the agent's prediction w.r.t. the target
glimpse.

## Prediction Target Space

The prediction target is a 2-dimensional `np.ndarray` containing the true coordinates of the target glimpse.

## Observation Space

The observation is a dictionary with the following keys:

| Key                | Type         | Description                                                                                                                                 |
|--------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `"glimpse"`        | `np.ndarray` | $G \times G \times C$ numpy array of type `float32` representing a glimpse of the image where each pixel is in the range $[-1, 1]$          |
| `"glimpse_pos"`    | `np.ndarray` | 2D numpy array of type `float32` containing the normalized position of the glimpse within the image in the range $[-1, 1]$                  |
| `"target_glimpse"` | `np.ndarray` | $G \times G \times C$ numpy array of type `float32` representing the target glimpse of the image where each pixel is in the range $[-1, 1]$ |
| `"time_step"`      | `float`      | The current time step between 0 and `image_perception_config.step_limit` normalized to the range $[-1, 1]$.                                 |

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \cdot{} \lVert action\rVert$.
- The negative mean squared error between the agent's prediction and the true coordinates of the target glimpse.

## Starting State

The glimpse's starting location and the target glimpse coordinates are chosen uniformly randomly within the image.

## Episode End

The episode ends with the terminate flag set when the maximum number of steps (`image_perception_config.step_limit`) is reached.

## Arguments

| Parameter                 | Type                                      | Default     | Description                                                                                                                             |
|---------------------------|-------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `image_perception_config` | `ap_gym.envs.image.ImagePerceptionConfig` |             | Configuration of the image perception environment. See the [ImagePerceptionConfig documentation](ImagePerceptionConfig.md) for details. |
| `render_mode`             | `Literal["rgb_array"]`                    | "rgb_array" | Rendering mode. Just "rgb_array" is supported currently.                                                                                |

## Overview of Implemented Environments

| Environment ID                           | Image type | # data points | Image size | Glimpse size | Image description                                                                                       |
|------------------------------------------|------------|---------------|------------|--------------|---------------------------------------------------------------------------------------------------------|
| [CIFAR10Loc-v0](CIFAR10Loc.md)           | RGB        | 50,000        | 32x32      | 5            | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).                 |
| [TinyImageNetLoc-v0](TinyImageNetLoc.md) | RGB        | 100,000       | 64x64      | 10           | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |
