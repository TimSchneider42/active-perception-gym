from __future__ import annotations

from abc import ABC
from typing import Generic, Any

import gymnasium as gym
import numpy as np

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
from .loss_fn import MSELossFn
from .types import ObsType, ActType, FullActType
from .log_wrapper import (
    ActivePerceptionLogWrapper,
    ActivePerceptionVectorLogWrapper,
)
import logging

logger = logging.getLogger(__name__)


def _make_mse_loss_fn_and_target_space(
    target_dim: int,
    prediction_low: float | None = None,
    prediction_high: float | None = None,
    target_std: float | None = None,
) -> tuple[MSELossFn, gym.spaces.Box]:
    prediction_target_space = gym.spaces.Box(
        low=prediction_low, high=prediction_high, shape=(target_dim,)
    )
    if target_std is not None:
        target_std = target_std
    elif np.isfinite(prediction_low) and np.isfinite(prediction_high):
        # Assume uniform distribution over the prediction target space
        target_std = (prediction_high - prediction_low) / np.sqrt(12)
    else:
        target_std = None
    loss_fn = MSELossFn(target_std=target_std)
    if target_std is not None:
        loss_fn = loss_fn.normalized
    else:
        logger.warning(
            "Prediction target space is unbounded, and target_std is not provided. MSE loss will not be normalized."
        )
    return loss_fn, prediction_target_space


class ActiveRegressionEnv(
    ActivePerceptionEnv[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        target_dim: int,
        inner_action_space: gym.Space[ActType],
        prediction_low: float | np.ndarray = -np.inf,
        prediction_high: float | np.ndarray = np.inf,
        target_std: float | None = None,
    ):
        prediction_space = gym.spaces.Box(
            low=prediction_low, high=prediction_high, shape=(target_dim,)
        )
        self.action_space = ActivePerceptionActionSpace(
            inner_action_space, prediction_space
        )
        self.loss_fn, self.prediction_target_space = _make_mse_loss_fn_and_target_space(
            target_dim, prediction_low, prediction_high, target_std
        )


class ActiveRegressionVectorEnv(
    ActivePerceptionVectorEnv[ObsType, ActType, np.ndarray, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def __init__(
        self,
        num_envs: int,
        target_dim: int,
        single_inner_action_space: gym.Space[ActType],
        prediction_low: float = -np.inf,
        prediction_high: float = np.inf,
        target_std: float | None = None,
    ):
        self.num_envs = num_envs
        single_prediction_space = gym.spaces.Box(
            low=prediction_low, high=prediction_high, shape=(target_dim,)
        )
        self.single_action_space = ActivePerceptionActionSpace(
            single_inner_action_space, single_prediction_space
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space, num_envs
        )
        self.loss_fn, self.single_prediction_target_space = (
            _make_mse_loss_fn_and_target_space(
                target_dim, prediction_low, prediction_high, target_std
            )
        )
        self.prediction_target_space = gym.vector.utils.batch_space(
            self.single_prediction_target_space, num_envs
        )


class ActiveRegressionLogWrapper(
    ActivePerceptionLogWrapper[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def _step_metrics(
        self, action: FullActType[ActType, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        residual = info["prediction"]["target"] - action["prediction"]
        return {
            "euclidean_distance": np.linalg.norm(residual),
            "mse": np.mean(residual**2),
        }


class ActiveRegressionVectorLogWrapper(
    ActivePerceptionVectorLogWrapper[ObsType, ActType, np.ndarray, np.ndarray],
    Generic[ObsType, ActType],
    ABC,
):
    def _step_metrics(
        self, action: FullActType[ActType, np.ndarray], info: dict[str, Any]
    ) -> dict[str, Any]:
        residual = info["prediction"]["target"] - action["prediction"]
        return {
            "euclidean_distance": np.linalg.norm(residual, axis=-1),
            "mse": np.mean(residual**2, axis=-1),
        }
