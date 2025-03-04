# Active Image Classification

In image classification environments, the agent has to classify an image by moving a small glimpse around the image.
The glimpse is never large enough to see the entire image at once, so the agent has to move around to gather information.

All image classification environments in _ap_gym_ share the following properties:

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
            <code>&nbsp;&nbsp;"glance": ImageSpace(width=G, height=G, channels=C, dtype=np.float32),</code><br>
            <code>&nbsp;&nbsp;"glance_pos": Box(-1.0, 1.0, shape=(2,), dtype=np.float32)</code><br>
            <code>})</code>
        </td>
    </tr>
</table>


where $K \in \mathbb{N}$ is the number of classes in the environment, $G \in \mathbb{N}$ is the glance size, and $C \in \mathbb{N}$ is the number of image channels (1 for grayscale, 3 for RGB).

## Action Space

The action is an `ndarray` with shape `(2,)` consisting of continuous values in the interval $[-1, 1]$.

- `action[0]`: Horizontal sensor movement
- `action[1]`: Vertical sensor movement

**Note**: Actions are normalized and scaled by the environment's `max_step_length`, which is 0.2 (20% of the image) by default.

## Observation Space

The observation is a dictionary with keys `"glimpse"` and `"glimpse_pos"`.
The glimpse is a $G \times G \times C$ `ndarray` representing a glimpse of the image where each pixel is in the range $[-1, 1]$.
The `"glimpse_pos"` is an `ndarray` with shape `(2,)` containing the normalized position of the glimpse within the image in the range $[-1, 1]$.

## Prediction Space

The prediction is a $K$ dimensional `ndarray` containing the logits of the agent's prediction w.r.t. the class label.
The prediction target is a scalar integer in the range $[0, K - 1]$, representing the true class.

## Rewards

The reward at each timestep is a sum of:

- A small action regularization equal to $10^{-3} \lVert action\rVert$.
- The negative cross-entropy loss between the agent's prediction and the true class.

## Starting State

The glimpse starts at a random position within the image.

## Episode End

The episode ends when the maximum number of steps (`max_episode_steps`, default: `16`) is reached.

## Overview of Implemented Environments

| Environment ID                      | Image type | Number of classes | Image size | Glimpse size | Image description                                                               |
|-------------------------------------|------------|-------------------|------------|--------------|---------------------------------------------------------------------------------|
| [CircleSquare-v0](CircleSquare)              | Grayscale  | 2                 | 28x28      | 5            | An image containing either a circle or square.                                  |
| [MNIST-v0](MNIST)                | Grayscale  | 10                | 28x28      | 5            | Handwritten digits from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). |