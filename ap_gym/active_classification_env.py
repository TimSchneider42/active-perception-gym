from __future__ import annotations

from abc import ABC
from typing import TypeVar, Generic, Any

import gymnasium as gym
import numpy as np
import scipy.special

from .active_perception_env import (
    ActivePerceptionEnv,
    LossFn,
    ActivePerceptionActionSpace,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorEnv,
    FullActType,
    PredType,
)

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class CrossEntropyLossFn(LossFn[np.ndarray, int | np.ndarray]):
    def numpy(
        self,
        prediction: np.ndarray,
        target: int | np.ndarray,
        batch_shape: tuple[int, ...] = (),
    ) -> float:
        return -np.take_along_axis(
            scipy.special.log_softmax(prediction, axis=-1), target[..., None], axis=-1
        )[..., 0]

    def torch(
        self,
        prediction: "torch.Tensor",
        target: int | "torch.Tensor",
        batch_shape: tuple[int, ...] = (),
    ) -> "torch.Tensor":
        return -torch.take_along_dim(
            torch.nn.functional.log_softmax(prediction, dim=-1),
            target[..., None],
            dim=-1,
        )[..., 0]

    def jax(
        self,
        prediction: "jax.Array",
        target: int | "jax.Array",
        batch_shape: tuple[int, ...] = (),
    ) -> "jax.Array":
        return -jnp.take_along_axis(
            jax.nn.log_softmax(prediction), target[..., None], axis=-1
        )[..., 0]


class ActiveClassificationEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, int],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(self, num_classes: int, inner_action_space: gym.Space[ActType]):
        prediction_space = gym.spaces.Box(-np.inf, np.inf, shape=(num_classes,))
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.prediction_target_space = gym.spaces.Discrete(num_classes)
        self.loss_fn = CrossEntropyLossFn()
        self.__current_correct_sum = None
        self.__current_step = None
        self.__first_correct = None
        self.__last_incorrect = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__current_correct_sum = 0
        self.__current_step = 0
        self.__first_correct = None
        self.__last_incorrect = 0
        return super().reset(seed=seed, options=options)

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        is_correct = self.current_prediction_target == action["prediction"].argmax(
            axis=-1
        )
        self.__current_correct_sum += is_correct
        self.__current_step += 1
        if not is_correct:
            self.__last_incorrect = self.__current_step
        elif self.__first_correct is None:
            self.__first_correct = self.__current_step
        if "stats" not in info:
            info["stats"] = {}
        done = terminated or truncated
        if done:
            info["stats"].update(
                {
                    "avg_accuracy": self.__current_correct_sum
                    / max(self.__current_step, 1),
                    "final_accuracy": float(is_correct),
                }
            )
            if self.__first_correct is not None:
                info["stats"]["first_correct"] = self.__first_correct
            info["stats"]["last_incorrect"] = self.__last_incorrect
        return obs, reward, terminated, truncated, info


class ActiveClassificationVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        num_envs: int,
        num_classes: int,
        single_inner_action_space: gym.Space[ActType],
        inner_action_space: gym.Space[ActType],
    ):
        self.num_envs = num_envs
        single_prediction_space = gym.spaces.Box(-np.inf, np.inf, shape=(num_classes,))
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        prediction_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(
                num_envs,
                num_classes,
            ),
        )
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.single_prediction_target_space = gym.spaces.Discrete(num_classes)
        self.prediction_target_space = gym.spaces.MultiDiscrete((num_envs, num_classes))
        self.loss_fn = CrossEntropyLossFn()
        self.__current_correct_sum = None
        self.__current_step = None
        self.__prev_done = None
        self.__first_correct = None
        self.__last_incorrect = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__current_correct_sum = np.zeros(self.num_envs, dtype=np.float32)
        self.__current_step = np.zeros(self.num_envs, dtype=np.int32)
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__first_correct = np.full(self.num_envs, -1, dtype=np.int32)
        self.__last_incorrect = np.zeros(self.num_envs, dtype=np.int32)
        return super().reset(seed=seed, options=options)

    def _step(
        self, action: ActType, prediction: np.ndarray
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
        obs, reward, terminated, truncated, info, label = super()._step(
            action, prediction
        )
        is_correct = label == action["prediction"].argmax(axis=-1)
        self.__current_correct_sum += is_correct
        self.__current_correct_sum[self.__prev_done] = 0
        self.__current_step[self.__prev_done] = 0
        self.__current_step[~self.__prev_done] += 1
        self.__first_correct[self.__prev_done] = -1
        self.__last_incorrect[self.__prev_done] = 0
        set_first_correct = (
            ~self.__prev_done & is_correct & (self.__first_correct == -1)
        )
        self.__first_correct[set_first_correct] = self.__current_step[set_first_correct]
        self.__last_incorrect[~is_correct] = self.__current_step[~is_correct]
        if "stats" not in info:
            info["stats"] = {}
        done = terminated | truncated
        if np.any(done):
            avg_accuracy = self.__current_correct_sum / np.maximum(
                self.__current_step, 1
            )
            info["stats"].update(
                {
                    "avg_accuracy": avg_accuracy,
                    "_avg_accuracy": done,
                    "final_accuracy": is_correct.astype(np.float32),
                    "_final_accuracy": done,
                    "first_correct": self.__first_correct,
                    "_first_correct": done & (self.__first_correct != -1),
                    "last_incorrect": self.__last_incorrect,
                    "_last_incorrect": done,
                }
            )
        self.__prev_done = done
        return obs, reward, terminated, truncated, info, label
