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
from .image import (
    HuggingfaceImageClassificationDataset,
    CircleSquareDataset,
    ImageClassificationDataset,
    ImagePerceptionConfig,
)


def register_image_classification_env(
    name: str,
    dataset: ImageClassificationDataset,
    max_episode_steps: int,
    kwargs: dict[str, Any] | None = None,
):
    if kwargs is None:
        kwargs = {}
    gym.envs.registration.register(
        id=name,
        kwargs=dict(
            image_perception_config=ImagePerceptionConfig(dataset=dataset, **kwargs)
        ),
        entry_point="ap_gym.envs.image_classification:ImageClassificationEnv",
        vector_entry_point="ap_gym.envs.image_classification:ImageClassificationVectorEnv",
        max_episode_steps=max_episode_steps,
    )


def register_envs():
    SIZES = {
        "": (28, 28),
        **{f"-s{s}": (s, s) for s in [10, 20, 28]},
    }

    SHOW_GRADIENT = {"": True, "-nograd": False}

    for size_suffix, size in SIZES.items():
        for sg_suffix, show_gradient in SHOW_GRADIENT.items():
            register_image_classification_env(
                name=f"CircleSquare{size_suffix}{sg_suffix}-v0",
                dataset=CircleSquareDataset(
                    image_shape=size, show_gradient=show_gradient
                ),
                max_episode_steps=16,
            )

    for split in ["train", "test"]:
        split_names = [f"-{split}"]
        if split == "train":
            split_names.append("")
        for split_name in split_names:
            register_image_classification_env(
                name=f"MNIST{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset("mnist", split=split),
                max_episode_steps=16,
            )

            render_kwargs = dict(
                render_overlay_base_color=(0, 0, 0, 128),
                render_good_color=(0, 255, 0, 60),
                render_bad_color=(255, 0, 0, 60),
            )
            register_image_classification_env(
                name=f"CIFAR10{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "cifar10", image_feature_name="img", split=split
                ),
                max_episode_steps=16,
                kwargs=render_kwargs,
            )

            register_image_classification_env(
                name=f"TinyImageNet{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "zh-plus/tiny-imagenet", split=split
                ),
                max_episode_steps=16,
                kwargs=dict(sensor_size=(10, 10), **render_kwargs),
            )

    gym.envs.registration.register(
        id="LightDark-v0",
        entry_point="ap_gym.envs.light_dark:LightDarkEnv",
        max_episode_steps=16,
    )

    gym.envs.registration.register(
        id="Localization2D-v0",
        entry_point="ap_gym.envs.localization2d:Localization2DEnv",
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
