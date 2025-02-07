from typing import Any, Union, Optional, Dict, Sequence, Callable

import gymnasium as gym

from .active_perception_env import (
    BaseActivePerceptionEnv,
    ActivePerceptionWrapper,
    ensure_active_perception_env,
)
from .active_perception_vector_env import (
    BaseActivePerceptionVectorEnv,
    ensure_active_perception_vector_env,
)
from .circle_square_env import CircleSquareEnv, CircleSquareVectorEnv
from .mnist_env import MNISTEnv, MNISTVectorEnv


def register_envs():
    SIZES = {
        "": (28, 28),
        **{f"-s{s}": (s, s) for s in [10, 20, 28]},
    }

    SHOW_GRADIENT = {"": True, "-ng": False}

    for size_suffix, size in SIZES.items():
        for sg_suffix, show_gradient in SHOW_GRADIENT.items():
            gym.envs.registration.register(
                id=f"CircleSquare{size_suffix}{sg_suffix}-v0",
                kwargs=dict(image_shape=size, show_gradient=show_gradient),
                entry_point=CircleSquareEnv,
                vector_entry_point=CircleSquareVectorEnv,
            )

    gym.envs.registration.register(
        id="MNIST-v0", entry_point=MNISTEnv, vector_entry_point=MNISTVectorEnv
    )


def make(
    id: Union[str, gym.envs.registration.EnvSpec],
    max_episode_steps: Optional[int] = None,
    disable_env_checker: Optional[bool] = None,
    **kwargs: Any,
) -> BaseActivePerceptionEnv:
    return ensure_active_perception_env(
        gym.make(
            id,
            max_episode_steps=max_episode_steps,
            disable_env_checker=disable_env_checker,
            **kwargs,
        )
    )


def make_vec(
    id: Union[str, gym.envs.registration.EnvSpec],
    num_envs: int = 1,
    vectorization_mode: Optional[Union[gym.VectorizeMode, str]] = None,
    vector_kwargs: Optional[Dict[str, Any]] = None,
    wrappers: Sequence[Callable[[BaseActivePerceptionEnv], ActivePerceptionWrapper]]
    | None = None,
    **kwargs,
) -> BaseActivePerceptionVectorEnv:
    return ensure_active_perception_vector_env(
        gym.make_vec(
            id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs
        )
    )
