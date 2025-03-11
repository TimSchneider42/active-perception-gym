# Image Classification Environments

In image classification environments, the agent has to classify an image by moving a small glimpse around the image.
The glimpse is never large enough to see the entire image at once, so the agent has to move around to gather
information.

Consider the following example from the [CIFAR10](CIFAR10.md) environment:
<p align="center"><img src="img/CIFAR10-v0.gif" alt="CIFAR10-v0" width="200px"/></p>

Marked in blue is the agent's current glimpse.
We mark the history of glimpses the agent has taken in a color scale ranging from red to green, red meaning that the
agent predicted a probability of 0 for the correct class and green meaning that the agent predicted a probability of 1
for the correct class.

All image classification environments in _ap_gym_ are instantiations of the
`ap_gym.envs.image_classification.ImageClassification` class and share the following properties:

## Properties

<table>
    <tr>
        <td><strong>Action Space</strong></td>
        <td><code>Box(-1.0, 1.0, shape=(2,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Space</strong></td>
        <td><code>Box(-inf, inf, shape=(K,), dtype=np.float32)</code></td>
    </tr>
    <tr>
        <td><strong>Prediction Target Space</strong></td>
        <td><code>Discrete(K)</code></td>
    </tr>
    <tr>
        <td><strong>Observation Space</strong></td>
        <td>
            <code>Dict({</code><br>
            <code>&nbsp;&nbsp;"glimpse": ImageSpace(width=G, height=G, channels=C, dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"glimpse_pos": Box(-1.0, 1.0, shape=(2,), dtype=np.float32)</code><br>
            <code>})</code>
        </td>
    </tr>
    <tr>
        <td><strong>Loss Function</strong></td>
        <td>
            <code>ap_gym.CrossEntropyLossFn()</code>
        </td>
    </tr>
</table>


where $K \in \mathbb{N}$ is the number of classes in the environment, $G \in \mathbb{N}$ is the glimpse size,
and $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB).

## Action Space

The action is an `np.ndarray` with shape `(2,)` consisting of continuous values in the interval $[-1, 1]$.

- `action[0]`: Horizontal sensor movement
- `action[1]`: Vertical sensor movement

**Note**: Actions are normalized and scaled by the environment's `max_step_length`, which is 0.2 (20% of the image) by
default.

## Prediction Space

The prediction is a $K$-dimensional `np.ndarray` containing the logits of the agent's prediction w.r.t. the class label.

## Prediction Target Space

The prediction target is a scalar integer in the range $[0, K - 1]$, representing the true class.

## Observation Space

The observation is a dictionary with keys `"glimpse"` and `"glimpse_pos"`.
The glimpse is a $G \times G \times C$ `np.ndarray` representing a glimpse of the image where each pixel is in the
range $[-1, 1]$.
The `"glimpse_pos"` is an `np.ndarray` with shape `(2,)` containing the normalized position of the glimpse within the
image in the range $[-1, 1]$.

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \cdot{} \lVert action\rVert$.
- The negative cross-entropy loss between the agent's prediction and the true class.

## Starting State

The glimpse starts at a uniformly random position within the image.

## Episode End

The episode ends when the maximum number of steps (`max_episode_steps`, default: `16`) is reached.

## Arguments

| Parameter                 | Type                                      | Default     | Description                                                                                                                             |
|---------------------------|-------------------------------------------|-------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| `image_perception_config` | `ap_gym.envs.image.ImagePerceptionConfig` |             | Configuration of the image perception environment. See the [ImagePerceptionConfig documentation](ImagePerceptionConfig.md) for details. |
| `render_mode`             | `Literal["rgb_array"]`                    | "rgb_array" | Rendering mode. Just "rgb_array" is supported currently.                                                                                |
| `max_episode_steps`       | `int`                                     | 16          | Maximum steps per episode.                                                                                                              |

## Overview of Implemented Environments

| Environment ID                     | Image type | # classes | # data points | Image size | Glimpse size | Image description                                                                                       |
|------------------------------------|------------|-----------|---------------|------------|--------------|---------------------------------------------------------------------------------------------------------|
| [CircleSquare-v0](CircleSquare.md) | Grayscale  | 2         | 1,568         | 28x28      | 5            | An image containing either a circle or square.                                                          |
| [MNIST-v0](MNIST.md)               | Grayscale  | 10        | 60,000        | 28x28      | 5            | Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).                         |
| [CIFAR10-v0](CIFAR10.md)           | RGB        | 10        | 50,000        | 32x32      | 5            | Natural images from the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).                 |
| [TinyImageNet-v0](TinyImageNet.md) | RGB        | 200       | 100,000       | 64x64      | 10           | Natural images from the [Tiny ImageNet dataset](https://huggingface.co/datasets/zh-plus/tiny-imagenet). |
