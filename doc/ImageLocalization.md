# Image Localization Environments

In image classification environments, the agent has to localize a given part of the image by moving a small glimpse
around.
The glimpse is never large enough to see the entire image at once, so the agent has to move around to gather
information.
Unlike the [Image Classification Environments](ImageClassification), this task is a regression task where the agent has
to predict the coordinates of the target glimpse it is provided.

Consider the following example from the [TinyImageNetLoc](TinyImageNetLoc) environment:
<p align="center"><img src="img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/></p>
Marked in blue is the agent's current glimpse.
The transparent purple box represents the target glimpse the agent has to predict the coordinates of and the opaque purple box is the agent's current prediction.
We further mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that the prediction it took at this step was far from the target and green meaning that the prediction was close to the target.

All image classification environments in _ap_gym_ are instantiations of the
`ap_gym.envs.image_classification.ImageLocalization` class and share the following properties:

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

The action is an `np.ndarray` with shape `(2,)` consisting of continuous values in the interval $[-1, 1]$.

- `action[0]`: Horizontal sensor movement
- `action[1]`: Vertical sensor movement

**Note**: Actions are normalized and scaled by the environment's `max_step_length`, which is 0.2 (20% of the image) by
default.

## Prediction Space

The prediction is a 2-dimensional `np.ndarray` containing the coordinates of the agent's prediction w.r.t. the target
glimpse.

## Prediction Target Space

The prediction target is a 2-dimensional `np.ndarray` containing the true coordinates of the target glimpse.

## Observation Space

The observation is a dictionary with keys `"glimpse"`, `"glimpse_pos"`, and `"target_glimpse"`.
The glimpse is a $G \times G \times C$ `np.ndarray` representing a glimpse of the image where each pixel is in the
range $[-1, 1]$.
The `"glimpse_pos"` is an `np.ndarray` with shape `(2,)` containing the normalized position of the glimpse within the
image in the range $[-1, 1]$.
The `"target_glimpse"` is a $G \times G \times C$ `np.ndarray` representing the target glimpse of the image where each
pixel is in the range $[-1, 1]$.

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \cdot{} \lVert action\rVert$.
- The negative mean squared error between the agent's prediction and the true coordinates of the target glimpse.

## Starting State

The glimpse's starting location and the target glimpse coordinates are chosen uniformly randomly within the image.

## Episode End

The episode ends when the maximum number of steps (`max_episode_steps`, default: `16`) is reached.

## Arguments

| Parameter                 | Type                                      | Default     | Description                                                                                                                            |
|---------------------------|-------------------------------------------|-------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `image_perception_config` | `ap_gym.envs.image.ImagePerceptionConfig` |             | Configuration of the image perception environment. See the [ImagePerceptionConfig documentation](ImagePerceptionConfig) for details. |
| `render_mode`             | `Literal["rgb_array"]`                    | "rgb_array" | Rendering mode. Just "rgb_array" is supported currently.                                                                               |
| `max_episode_steps`       | `int`                                     | 16          | Maximum steps per episode.                                                                                                             |

## Overview of Implemented Environments

| Environment ID                        | Image type | # data points | Image size | Glimpse size | Image description                                                                                       |
|---------------------------------------|------------|---------------|------------|--------------|---------------------------------------------------------------------------------------------------------|
| [CIFAR10Loc-v0](CIFAR10Loc)           | RGB        | 50,000        | 32x32      | 5            | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).                 |
| [TinyImageNetLoc-v0](TinyImageNetLoc) | RGB        | 100,000       | 64x64      | 10           | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |
