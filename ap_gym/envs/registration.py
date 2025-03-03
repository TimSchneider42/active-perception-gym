from __future__ import annotations

from typing import Any, Sequence, Callable

import gymnasium as gym

from ap_gym import (
    BaseActivePerceptionEnv,
    ActivePerceptionWrapper,
    ensure_active_perception_env,
    BaseActivePerceptionVectorEnv,
    ensure_active_perception_vector_env,
)


def register_envs():
    SIZES = {
        "": (28, 28),
        **{f"-s{s}": (s, s) for s in [10, 20, 28]},
    }

    SHOW_GRADIENT = {"": True, "-nograd": False}

    for size_suffix, size in SIZES.items():
        for sg_suffix, show_gradient in SHOW_GRADIENT.items():
            gym.envs.registration.register(
                id=f"CircleSquare{size_suffix}{sg_suffix}-v0",
                kwargs=dict(image_shape=size, show_gradient=show_gradient),
                entry_point="ap_gym.envs.circle_square:CircleSquareEnv",
                vector_entry_point="ap_gym.envs.circle_square:CircleSquareVectorEnv",
                max_episode_steps=16,
            )

    gym.envs.registration.register(
        id="MNIST-v0",
        entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationEnv",
        vector_entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationVectorEnv",
        kwargs=dict(dataset_name="mnist"),
        max_episode_steps=16,
    )

    gym.envs.registration.register(
        id="CIFAR10-v0",
        entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationEnv",
        vector_entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationVectorEnv",
        kwargs=dict(dataset_name="cifar10", image_feature_name="img"),
        max_episode_steps=16,
    )

    gym.envs.registration.register(
        id="TinyImageNet-v0",
        entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationEnv",
        vector_entry_point="ap_gym.envs.huggingface_image_classification:HuggingfaceImageClassificationVectorEnv",
        kwargs=dict(dataset_name="zh-plus/tiny-imagenet", sensor_size=(10, 10)),
        max_episode_steps=16,
    )

    gym.envs.registration.register(
        id="LightDark-v0",
        entry_point="ap_gym.envs.light_dark:LightDarkEnv",
        max_episode_steps=16,
    )


def make(
    id: str | gym.envs.registration.EnvSpec,
    max_episode_steps: int | None = None,
    disable_env_checker: bool | None = None,
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
    id: str | gym.envs.registration.EnvSpec,
    num_envs: int = 1,
    vectorization_mode: gym.VectorizeMode | str | None = None,
    vector_kwargs: dict[str, Any | None] = None,
    wrappers: Sequence[Callable[[BaseActivePerceptionEnv], ActivePerceptionWrapper]]
    | None = None,
    **kwargs,
) -> BaseActivePerceptionVectorEnv:
    return ensure_active_perception_vector_env(
        gym.make_vec(
            id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs
        )
    )
