# Active Perception Gym

Extension of [Gymnasium](https://github.com/Farama-Foundation/Gymnasium/) for active perception tasks.

## Installation

This package can be installed using pip:

```bash
pip install ap_gym[OPTIONS]
```

where OPTIONS can be empty or `examples`, which installs dependencies for the examples.

## Basic Usage

_ap_gym_ adds functionality for active perception tasks to Gymnasium.
This guide assumes that you are familiar with Gymnasium, otherwise, please check out
their [documentation](https://gymnasium.farama.org/index.html).

### Active Perception

In the active perception domain, an agent's main objective is to gather information and make predictions about a desired
property of the environment.
Examples of such properties could be the location of an object in case of a search task or the class of an object the
agent in case of a classification task.
To gather information, the agent must interact with the environment, e.g. by moving a glimpse around in case of
the [CircleSquare](doc/CircleSquare) and [MNIST](doc/MNIST) tasks.

_ap_gym_ models active perception tasks as episodic processes in a way that is fully compatible to Gymnasium.
Each task is defined as a Gymnasium environment, in which the agent is additionally provided with a differentiable loss
function.
The purpose of the loss function is to provide the agent with a generalizable notion of the distance between its current
property prediction and the ground truth property.

In every episode, the agent may take a task-dependent number of steps to gather information.
Just like in Gymnasium, in every step the environment provides the agent with an observation, typically consisting of
scalar and/or image data.
In return, the agent must provide the environment with an action and a property prediction in every step.
Based on the action and prediction of the agent, the environment computes a reward in every step, which is the sum of a
regular RL reward (the base reward) and the negative value of the environment's loss function.
Hence, the agent has to make a prediction in every step, encouraging it to gather information quickly to maximize its
prediction reward early on.

### Environment Base Classes

Every task in _ap_gym_ is modeled as a subclass of `ap_gym.ActivePerceptionEnv` or `ap_gym.ActivePerceptionVectorEnv`.
`ap_gym.ActivePerceptionEnv` and `ap_gym.ActivePerceptionVectorEnv` subclass `gymnasium.Env` and
`gymnasium.vector.VectorEnv`, respectively.
Both subclasses extend their Gymnasium interfaces by four fields:

- `loss_fn`: The loss function of the environment. See [`ap_gym.LossFn`](#ap_gymlossfn).
- `prediction_space`: A `gymnasium.spaces.Space` defining the set of valid prediction values.
- `prediction_target_space`: A `gymnasium.spaces.Space` defining the set of valid prediction target values.
- `inner_action_space`: A `gymnasium.spaces.Space` defining the set of valid inner action values.
  Additionally, `ap_gym.ActivePerceptionVectorEnv` adds the respective single variants of the latter two fields:
  `single_prediction_space` and `single_inner_action_space`.

`ap_gym.ActivePerceptionEnv` and `ap_gym.ActivePerceptionVectorEnv` further enforce the agent's action space to be of
the following form:

```python
{
    "action": action,
    "prediction": prediction
}
```

where the set of valid `action` values is defined by the `inner_action_space` field of the respective environment, and
the set of valid `prediction` values is defined by the `prediction_space` field.

The info dictionary returned by the reset and step functions always contains the current prediction target in
`info["prediction"]["target"]`.
Additionally, the info dictionary returned by the step function contains the base reward (the reward without the
prediction loss) in `info["base_reward"]` and the prediction loss in `info["prediction"]["loss"]`.

To get an understanding of how this class is used, refer to the examples in the _examples_ directory and to
the [tasks](#tasks) defined by _ap_gym_.

### Loss Functions

The `ap_gym.LossFn` base class provides a differentiable implementation of the loss function for PyTorch and JAX.
`ap_gym.LossFn` has three functions: `numpy`, `torch`, and `jax`.
Each of these functions is the respective implementation of the loss function in Numpy, PyTorch, and JAX.
Note that only the PyTorch and JAX variant provide gradients as Numpy does not support autograd.

The signature of each framework-specific function is

```python
def fn(prediction: ArrayType, target: ArrayType, batch_shape: Tuple[int, ...] = ()) -> ArrayType:
    ...
```

where ArrayType is one of `np.ndarray`, `torch.Tensor`, or `jax.Array`.
`batch_shape` is used to specify the batch dimensions in case of a batched evaluation of the loss function, e.g.:

```python
loss = ap_gym.CrossEntropyLossFn()(np.zeros((3, 7, 10)), np.zeros((3, 7), dtype=np.int_), (3, 7))
```

### Representation of Image Observations

To help the agent differentiate between scalar and image observations, _ap_gym_ introduces a new type of Gymnasium
space: `ap_gym.ImageSpace`.
`ap_gym.ImageSpace` is a subclass of `gymnasium.spaces.Box` with some image specific convenience properties like
`width`, `height`, and `channels`.
Its main purpose, though, is to let the agent know that it has to interpret this part of the observation space as an
image.

### Using Gymnasium Wrappers

_ap_gym_ provides a method for using regular Gymnasium wrappers on `ap_gym.ActivePerceptionEnv` and
`ap_gym.ActivePerceptionVectorEnv` instances.
The issue with using Gymnasium wrappers naively is that the special fields `loss_fn`, `prediction_space`,
`prediction_target_space`, and `inner_action_space` do not get mapped through.
Hence,

```python
gymnasium.wrappers.TimeLimit(ap_gym.make("CircleSquare-v0"), 8).loss_fn
```

throws

```
AttributeError: 'TimeLimit' object has no attribute 'loss_fn'
```

To address this issue, `ap_gym.ensure_active_perception_env` and `ap_gym.ensure_active_perception_vector_env` can be
used:

```python
ap_gym.ActivePerceptionRestoreWrapper(gymnasium.wrappers.TimeLimit(ap_gym.make("CircleSquare-v0"), 8)).loss_fn
```

`ap_gym.ActivePerceptionRestoreWrapper` and `ap_gym.ActivePerceptionVectorRestoreWrapper` recursively traverse wrappers
until they find an active perception environment and map the special fields through.
Additionally, aside of Gymnasium wrappers, `ap_gym.ActivePerceptionVectorRestoreWrapper` also supports
`gymnasium.vector.SyncVectorEnv` and `gymnasium.vector.AsyncVectorEnv` and will restore proper vector versions of all
spaces if active perception environments are vectorized this way.

### Environments

TODO: document test environments
TOOD: document other environments (light dark, cifar10, TinyImageNet)

_ap_gym_ currently comes with a number of image classification environments, which are described in detail in
the [image classification documentation](doc/ImageClassification).

#### Image Classification

<table align="center" style="border-collapse: collapse; border: none;">
    <tr style="border: none;">
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/CircleSquare.md">
                <img src="doc/img/CircleSquare-v0.gif" alt="CircleSquare-v0" width="200px"/>
                <br/>
                CircleSquare-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/MNIST.md">
                <img src="doc/img/MNIST-v0.gif" alt="MNIST-v0" width="200px"/>
                <br/>
                MNIST-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/TinyImageNet.md">
                <img src="doc/img/TinyImageNet-v0.gif" alt="TinyImageNet-v0" width="200px"/>
                <br/>
                TinyImageNet-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/CIFAR10.md">
                <img src="doc/img/CIFAR10-v0.gif" alt="CIFAR10-v0" width="200px"/>
                <br/>
                CIFAR10-v0
            </a>
        </td>
    </tr>
</table>

#### 2D Localization

<table align="center" style="border-collapse: collapse; border: none;">
    <tr style="border: none;">
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/LightDark.md">
                <img src="doc/img/LightDark-v0.gif" alt="LightDark-v0" width="200px"/>
                <br/>
                LightDark-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/LIDARLocMaze.md">
                <img src="doc/img/LIDARLocMaze-v0.gif" alt="LIDARLocMaze-v0" width="200px"/>
                <br/>
                LIDARLocMaze-v0
            </a>
        </td>
    </tr>
</table>

#### Image Localization

<table align="center" style="border-collapse: collapse; border: none;">
    <tr style="border: none;">
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/TinyImageNetLoc.md">
                <img src="doc/img/TinyImageNetLoc-v0.gif" alt="TinyImageNetLoc-v0" width="200px"/>
                <br/>
                TinyImageNetLoc-v0
            </a>
        </td>
        <td align="center" style="border: none; padding: 10px;">
            <a href="doc/CIFAR10Loc.md">
                <img src="doc/img/CIFAR10Loc-v0.gif" alt="CIFAR10Loc-v0" width="200px"/>
                <br/>
                CIFAR10Loc-v0
            </a>
        </td>
    </tr>
</table>

### Converting Regular Gymnasium Environments to Active Perception Environments

It is possible to convert regular Gymnasium environments into a pseudo active perception environments with the
`ap_gym.PseudoActivePerceptionWrapper` and `ap_gym.PseudoActivePerceptionVectorWrapper`, respectively:

```python
env = gymnasium.make("CartPole-v1")
ap_env = ap_gym.PseudoActivePerceptionWrapper(env)
```

`ap_gym.PseudoActivePerceptionWrapper` and `ap_gym.PseudoActivePerceptionVectorWrapper` take the environment and add a
constant zero loss function as well as empty prediction and prediction target spaces.
The purpose of this conversion is to simplify testing of _ap_gym_ compatible algorithms on regular Gynmasium tasks.

If you want to support arbitrary Gymnasium and _ap_gym_ environments, use the `ap_gym.ensure_active_perception_env` and
`ap_gym.ensure_active_perception_vector_env` functions:

```python
ap_env_1 = ap_gym.ensure_active_perception_env(gymnasium.make("CartPole-v1"))
ap_env_2 = ap_gym.ensure_active_perception_env(ap_gym.make("CircleSquare-v0"))
ap_env_3 = ap_gym.ensure_active_perception_env(gymnasium.wrappers.TimeLimit(ap_gym.make("CircleSquare-v0"), 8))
```

These functions automatically detect whether to do nothing, apply a restoration wrapper, or perform pseudo active
perception environment conversion.

## Advanced Usage
For more advanced usage, i.e., defining custom environments or wrappers, refer to the [advanced usage documentation](doc/advanced_usage.md).
