from __future__ import annotations

from functools import partial, reduce
from typing import Any, Sequence, Callable

import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec

from ap_gym import (
    BaseActivePerceptionEnv,
    ActivePerceptionWrapper,
    ensure_active_perception_env,
    BaseActivePerceptionVectorEnv,
    ensure_active_perception_vector_env,
    ActiveRegressionLogWrapper,
    ActiveRegressionVectorLogWrapper,
    ActiveClassificationLogWrapper,
    ActiveClassificationVectorLogWrapper,
)
from .floor_map import FloorMapDatasetRooms, FloorMapDatasetMaze, FloorMapDataset
from .image import (
    HuggingfaceImageClassificationDataset,
    CircleSquareDataset,
    ImageClassificationDataset,
    ImagePerceptionConfig,
)
from .image_classification import ImageClassificationEnv, ImageClassificationVectorEnv
from .image_localization import ImageLocalizationEnv, ImageLocalizationVectorEnv

ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC = WrapperSpec(
    "ActiveRegressionLogWrapper", "ap_gym:ActiveRegressionLogWrapper", kwargs={}
)


def register_image_env(
    name: str,
    entry_point: Callable[[...], gym.Env],
    vector_entry_point: Callable[[...], gym.vector.VectorEnv],
    dataset: ImageClassificationDataset,
    step_limit: int,
    kwargs: dict[str, Any] | None = None,
    single_wrappers: tuple[Callable[[gym.Env], gym.Env]] = (),
    vector_wrappers: tuple[Callable[[gym.vector.VectorEnv], gym.vector.VectorEnv]] = (),
):
    if kwargs is None:
        kwargs = {}
    gym.register(
        id=name,
        kwargs=dict(
            image_perception_config=ImagePerceptionConfig(
                dataset=dataset, step_limit=step_limit, **kwargs
            )
        ),
        entry_point=lambda *args, **kwargs: reduce(
            lambda env, wrapper_fn: wrapper_fn(env),
            single_wrappers,
            entry_point(*args, **kwargs),
        ),
        vector_entry_point=lambda *args, **kwargs: reduce(
            lambda env, wrapper_fn: wrapper_fn(env),
            vector_wrappers,
            vector_entry_point(*args, **kwargs),
        ),
    )


register_image_classification_env = partial(
    register_image_env,
    entry_point=ImageClassificationEnv,
    vector_entry_point=ImageClassificationVectorEnv,
    single_wrappers=(ActiveClassificationLogWrapper,),
    vector_wrappers=(ActiveClassificationVectorLogWrapper,),
)

register_image_localization_env = partial(
    register_image_env,
    entry_point=ImageLocalizationEnv,
    vector_entry_point=ImageLocalizationVectorEnv,
    single_wrappers=(ActiveRegressionLogWrapper,),
    vector_wrappers=(ActiveRegressionVectorLogWrapper,),
)


def mk_time_limit(step_limit: int, issue_termination=False) -> WrapperSpec:
    return WrapperSpec(
        "TimeLimit",
        "ap_gym:TimeLimit",
        kwargs=dict(max_episode_steps=step_limit, issue_termination=issue_termination),
    )


def register_lidar_localization_env(
    name: str, dataset: FloorMapDataset, static_map: bool = False, step_limit: int = 100
):
    gym.register(
        id=name,
        entry_point="ap_gym.envs.lidar_localization2d:LIDARLocalization2DEnv",
        kwargs=dict(dataset=dataset, static_map=static_map),
        additional_wrappers=(
            mk_time_limit(step_limit),
            ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC,
        ),
    )


def register_envs():
    SIZES = {
        "": (28, 28),
        **{f"-s{s}": (s, s) for s in [15, 20, 28]},
    }

    SHOW_GRADIENT = {"": True, "-nograd": False}

    for size_suffix, size in SIZES.items():
        for sg_suffix, show_gradient in SHOW_GRADIENT.items():
            register_image_classification_env(
                name=f"CircleSquare{size_suffix}{sg_suffix}-v0",
                dataset=CircleSquareDataset(
                    image_shape=size, show_gradient=show_gradient
                ),
                step_limit=16,
            )

    image_env_render_kwargs = dict(
        render_unvisited_opacity=0.5,
        render_visited_opacity=0.25,
    )

    for split in ["train", "test"]:
        split_names = [f"-{split}"]
        if split == "train":
            split_names.append("")
        for split_name in split_names:
            register_image_classification_env(
                name=f"MNIST{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "mnist", channels=1, split=split
                ),
                step_limit=16,
            )

            register_image_classification_env(
                name=f"CIFAR10{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "cifar10", image_feature_name="img", split=split
                ),
                step_limit=16,
                kwargs=image_env_render_kwargs,
            )

            register_image_classification_env(
                name=f"TinyImageNet{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "zh-plus/tiny-imagenet",
                    split=split if split == "train" else "valid",
                ),
                step_limit=16,
                kwargs=dict(sensor_size=(10, 10), **image_env_render_kwargs),
            )

            register_image_localization_env(
                name=f"CIFAR10Loc{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "cifar10", image_feature_name="img", split=split
                ),
                step_limit=16,
                kwargs=image_env_render_kwargs,
            )

            register_image_localization_env(
                name=f"TinyImageNetLoc{split_name}-v0",
                dataset=HuggingfaceImageClassificationDataset(
                    "zh-plus/tiny-imagenet",
                    split=split if split == "train" else "valid",
                ),
                step_limit=16,
                kwargs=dict(sensor_size=(10, 10), **image_env_render_kwargs),
            )

    gym.register(
        id="LightDark-v0",
        entry_point="ap_gym.envs.light_dark:LightDarkEnv",
        additional_wrappers=(
            mk_time_limit(50),
            ACTIVE_REGRESSION_LOGGER_WRAPPER_SPEC,
        ),
    )

    register_lidar_localization_env(
        "LIDARLocMazeStatic-v0",
        dataset=FloorMapDatasetMaze(),
        static_map=True,
    )

    register_lidar_localization_env(
        "LIDARLocMaze-v0",
        dataset=FloorMapDatasetMaze(),
    )

    register_lidar_localization_env(
        "LIDARLocRoomsStatic-v0",
        dataset=FloorMapDatasetRooms(),
        static_map=True,
    )

    register_lidar_localization_env(
        "LIDARLocRooms-v0",
        dataset=FloorMapDatasetRooms(),
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
    wrappers: (
        Sequence[Callable[[BaseActivePerceptionEnv], ActivePerceptionWrapper]] | None
    ) = None,
    **kwargs,
) -> BaseActivePerceptionVectorEnv:
    return ensure_active_perception_vector_env(
        gym.make_vec(
            id, num_envs, vectorization_mode, vector_kwargs, wrappers, **kwargs
        )
    )
