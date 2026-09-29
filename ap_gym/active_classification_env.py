from __future__ import annotations

from abc import ABC
from collections import deque
from typing import Generic, Any

import gymnasium as gym
import numpy as np
import scipy

from .active_perception_env import (
    ActivePerceptionEnv,
    ActivePerceptionActionSpace,
    ActivePerceptionWrapper,
)
from .active_perception_vector_env import (
    ActivePerceptionVectorEnv,
    ActivePerceptionVectorWrapper,
    FullActType,
    PredType,
)
from .logit_space import LogitSpace
from .loss_fn import CrossEntropyLossFn
from .types import ObsType, ActType, FullActType
from .log_wrapper import (
    ActivePerceptionLogWrapper,
    ActivePerceptionVectorLogWrapper,
)


class ActiveClassificationEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, int],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(self, num_classes: int, inner_action_space: gym.Space[ActType]):
        prediction_space = LogitSpace(-np.inf, np.inf, shape=(num_classes,))
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.prediction_target_space = gym.spaces.Discrete(num_classes)
        self.loss_fn = CrossEntropyLossFn(num_classes=num_classes).normalized


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
    ):
        self.num_envs = num_envs
        single_prediction_space = LogitSpace(-np.inf, np.inf, shape=(num_classes,))
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.single_prediction_target_space = gym.spaces.Discrete(num_classes)
        self.prediction_target_space = gym.spaces.MultiDiscrete(
            [num_classes] * num_envs
        )
        self.loss_fn = CrossEntropyLossFn(num_classes=num_classes).normalized


class ActiveClassificationLogWrapper(
    ActivePerceptionLogWrapper[ObsType, ActType, np.ndarray, int],
    Generic[ObsType, ActType],
    ABC,
):
    def _step_metrics(
        self, action: FullActType[ActType, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        prob = float(
            scipy.special.softmax(action["prediction"])[info["prediction"]["target"]]
        )
        return {
            "correct_label_prob": prob,
            "accuracy": float(prob > 1 / self.prediction_target_space.n),
        }

    def _episode_metrics(self, metrics: dict[str, deque]) -> dict[str, Any]:
        is_correct = np.asarray(metrics["accuracy"], dtype=np.bool_)
        stats = {}
        first_correct = np.where(is_correct)[0]
        if len(first_correct) > 0:
            stats["first_correct"] = first_correct[0]
        last_incorrect = np.where(~is_correct)[0]
        if len(last_incorrect) > 0:
            stats["last_incorrect"] = last_incorrect[-1]
        return stats


class ActiveClassificationVectorLogWrapper(
    ActivePerceptionVectorLogWrapper[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def _step_metrics(
        self, action: FullActType[ActType, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        probs = scipy.special.softmax(action["prediction"], axis=-1)
        prob = probs[np.arange(self.num_envs), info["prediction"]["target"]]
        num_classes = self.single_prediction_target_space.n
        return {
            "correct_label_prob": prob.astype(np.float32),
            "accuracy": (prob > 1 / num_classes).astype(np.float32),
        }

    def _episode_metrics(self, metrics: dict[str, tuple[deque, ...]]) -> dict[str, Any]:
        first_correct = np.full(self.num_envs, -1, dtype=np.int32)
        first_correct_valid = np.zeros(self.num_envs, dtype=np.bool_)
        last_incorrect = np.full(self.num_envs, -1, dtype=np.int32)
        last_incorrect_valid = np.zeros(self.num_envs, dtype=np.bool_)
        for i in range(self.num_envs):
            is_correct = np.asarray(metrics["accuracy"][i], dtype=np.bool_)
            candidates = np.where(is_correct)[0]
            if len(candidates) > 0:
                first_correct[i] = candidates[0]
                first_correct_valid[i] = True
            candidates = np.where(~is_correct)[0]
            if len(candidates) > 0:
                last_incorrect[i] = candidates[-1]
                last_incorrect_valid[i] = True
        return {
            "first_correct": first_correct,
            "_first_correct": first_correct_valid,
            "last_incorrect": last_incorrect,
            "_last_incorrect": last_incorrect_valid,
        }
