from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from typing import Any, Generic, Iterator, Callable, Sequence

import gymnasium as gym
import numpy as np

from ap_gym.active_perception_env import find_loss_and_pred_space
from .active_perception_env import (
    ActivePerceptionActionSpace,
    LossFn,
    BaseActivePerceptionEnv,
    NoActivePerceptionEnvError,
)
from .loss_fn import ZeroLossFn
from .util import update_info_metrics_vec
from .types import (
    ObsType,
    ActType,
    PredType,
    PredTargetType,
    ArrayType,
    FullActType,
    WrapperObsType,
    WrapperActType,
    WrapperPredType,
    WrapperPredTargetType,
    WrapperArrayType,
)

BaseActivePerceptionEnvType = BaseActivePerceptionEnv[
    ObsType, ActType, PredType, PredTargetType
]
EnvFnsType = (
    Iterator[Callable[[], BaseActivePerceptionEnvType]]
    | Sequence[Callable[[], BaseActivePerceptionEnvType]]
)


class BaseActivePerceptionVectorEnv(
    gym.vector.VectorEnv[ObsType, FullActType[ActType, PredType], ArrayType],
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
    ABC,
):
    # Set these in every subclass
    prediction_target_space: gym.Space[PredTargetType]
    single_prediction_target_space: gym.Space[PredTargetType]
    action_space: ActivePerceptionActionSpace[ActType, PredType]
    single_action_space: ActivePerceptionActionSpace[ActType, PredType]
    loss_fn: LossFn[PredType, PredTargetType]

    @property
    def prediction_space(self) -> gym.Space[PredType]:
        return self.action_space["prediction"]

    @property
    def inner_action_space(self) -> gym.Space[ActType]:
        return self.action_space["action"]

    @property
    def single_prediction_space(self) -> gym.Space[PredType]:
        return self.single_action_space["prediction"]

    @property
    def single_inner_action_space(self) -> gym.Space[ActType]:
        return self.single_action_space["action"]


class ActivePerceptionVectorEnv(
    BaseActivePerceptionVectorEnv[
        ObsType, ActType, PredType, PredTargetType, ArrayType
    ],
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
    ABC,
):
    __prev_done: np.ndarray | None = None
    __prediction_metrics: "dict[str, tuple[deque, ...]] | None" = None

    @abstractmethod
    def _step(
        self, action: ActType, prediction: PredType
    ) -> tuple[
        ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any], PredTargetType
    ]:
        pass

    def step(
        self, action: FullActType[ActType, PredType]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        (
            obs,
            base_reward,
            terminated,
            truncated,
            info,
            prediction_target,
        ) = self._step(action["action"], action["prediction"])

        batch_shape = (self.num_envs,) if isinstance(self, gym.vector.VectorEnv) else ()
        prediction_loss, prediction_metrics = self.loss_fn.numpy_loss_and_metrics(
            action["prediction"], prediction_target, batch_shape, rng=self.np_random
        )

        info.update(
            {
                "base_reward": base_reward,
                "prediction": {
                    "target": prediction_target,
                    "loss": prediction_loss,
                },
            }
        )

        done = terminated | truncated
        self.__accumulate_prediction_metrics(prediction_metrics, done)
        if np.any(done):
            info = update_info_metrics_vec(info, self.__prediction_metrics, done)

        return obs, base_reward - prediction_loss, terminated, truncated, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__prev_done = np.zeros(self.num_envs, dtype=np.bool_)
        self.__prediction_metrics = defaultdict(
            lambda: tuple(deque() for _ in range(self.num_envs))
        )
        return super().reset(seed=seed, options=options)

    def __accumulate_prediction_metrics(
        self, metrics: dict[str, np.ndarray], done: np.ndarray
    ) -> None:
        """
        Append this step's per-sample metrics, keyed per environment.

        A ``_<name>`` entry is the per-sample validity mask of ``<name>`` and is
        accumulated like any other metric; update_info_metrics_vec interprets it.

        Unlike the single environment, sub-environments restart individually without a
        reset(), so each one's accumulator is cleared on its own previous done.
        """
        for name, values in metrics.items():
            per_env = self.__prediction_metrics[name]
            for i in range(self.num_envs):
                if self.__prev_done[i]:
                    per_env[i].clear()
                per_env[i].append(values[i])
        self.__prev_done = done


class ActivePerceptionVectorWrapper(
    gym.vector.VectorWrapper,  # This thing is not generic for some reason
    BaseActivePerceptionVectorEnv[
        WrapperObsType,
        WrapperActType,
        WrapperPredType,
        WrapperPredTargetType,
        WrapperArrayType,
    ],
    Generic[
        WrapperObsType,
        WrapperActType,
        WrapperPredType,
        WrapperPredTargetType,
        WrapperArrayType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
    ],
):
    env: BaseActivePerceptionVectorEnv[
        ObsType, ActType, PredType, PredTargetType, ArrayType
    ]

    def __init__(
        self,
        env: gym.vector.VectorEnv[ObsType, FullActType[ActType, PredType], ArrayType],
    ):
        env = ensure_active_perception_vector_env(env)
        self._action_space: ActivePerceptionActionSpace[ActType, PredType] | None
        self._single_action_space: ActivePerceptionActionSpace[ActType, PredType] | None
        self._prediction_target_space: gym.Space[WrapperPredTargetType] | None = None
        self._single_prediction_target_space: (
            gym.Space[WrapperPredTargetType] | None
        ) = None
        self._loss_fn: LossFn[WrapperPredType, WrapperPredTargetType] | None = None
        super().__init__(env)

    @property
    def loss_fn(self) -> LossFn[WrapperPredType, WrapperPredTargetType]:
        if self._loss_fn is not None:
            return self._loss_fn
        return self.env.loss_fn

    @property
    def prediction_target_space(self) -> WrapperPredTargetType:
        if self._prediction_target_space is not None:
            return self._prediction_target_space
        return self.env.prediction_target_space

    @property
    def single_prediction_target_space(self) -> WrapperPredTargetType:
        if self._single_prediction_target_space is not None:
            return self._single_prediction_target_space
        return self.env.single_prediction_target_space


class PseudoActivePerceptionVectorWrapper(
    BaseActivePerceptionVectorEnv[ObsType, ActType, tuple[()], tuple[()], np.ndarray],
    gym.vector.VectorWrapper,
    Generic[ObsType, ActType],
):
    def __init__(self, env: gym.vector.VectorEnv[ObsType, ActType, np.ndarray]):
        gym.vector.VectorWrapper.__init__(self, env)
        self.single_action_space = ActivePerceptionActionSpace(
            self.env.single_action_space, gym.spaces.Tuple(())
        )
        self.action_space = ActivePerceptionActionSpace(
            self.env.action_space, gym.spaces.Tuple(())
        )
        self.single_prediction_target_space = gym.spaces.Tuple(())
        self.prediction_target_space = gym.spaces.Tuple(())
        self.loss_fn = ZeroLossFn()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any | None] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info["prediction"] = {
            "target": (),
        }
        return obs, info

    def step(
        self, action: FullActType[ActType, None]
    ) -> tuple[ObsType, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action["action"])
        info.update(
            {
                "base_reward": reward,
                "prediction": {
                    "target": (),
                    "loss": np.zeros(self.num_envs, dtype=np.float32),
                },
            }
        )
        return obs, reward, terminated, truncated, info


def find_loss_and_pred_space_vec(
    env: gym.vector.VectorEnv,
) -> tuple[
    LossFn[PredType, PredTargetType],
    gym.Space[PredTargetType],
    gym.Space[PredTargetType],
]:
    if isinstance(env, BaseActivePerceptionVectorEnv):
        return (
            env.loss_fn,
            env.single_prediction_target_space,
            env.prediction_target_space,
        )
    else:
        if isinstance(env, gym.vector.VectorWrapper):
            return find_loss_and_pred_space_vec(env.env)
        elif isinstance(env, gym.vector.SyncVectorEnv):
            loss_fn, single_pred_space = find_loss_and_pred_space(env.envs[0])
            return (
                loss_fn,
                single_pred_space,
                gym.vector.utils.batch_space(single_pred_space, env.num_envs),
            )
        elif isinstance(env, gym.vector.AsyncVectorEnv):
            dummy_env = env.env_fns[0]()
            loss_fn, single_pred_space = find_loss_and_pred_space(dummy_env)
            dummy_env.close()
            del dummy_env
            return (
                loss_fn,
                single_pred_space,
                gym.vector.utils.batch_space(single_pred_space, env.num_envs),
            )
        else:
            raise NoActivePerceptionEnvError(
                "The environment does not contain an ActivePerceptionEnv"
            )


class ActivePerceptionVectorRestoreWrapper(
    gym.vector.VectorWrapper,
    BaseActivePerceptionVectorEnv[
        ObsType, ActType, PredType, PredTargetType, ArrayType
    ],
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
):
    def __init__(self, env: gym.vector.VectorEnv):
        super().__init__(env)
        (
            self.__loss_fn,
            self.__single_prediction_target_space,
            self.__prediction_target_space,
        ) = find_loss_and_pred_space_vec(env)
        act_space = super().action_space
        self.__action_space = ActivePerceptionActionSpace(
            act_space["action"], act_space["prediction"], seed=act_space._np_random
        )
        single_act_space = super().single_action_space
        self.__single_action_space = ActivePerceptionActionSpace(
            single_act_space["action"],
            single_act_space["prediction"],
            seed=single_act_space._np_random,
        )

    @property
    def loss_fn(self) -> LossFn[WrapperPredType, WrapperPredTargetType]:
        return self.__loss_fn

    @property
    def action_space(self) -> ActivePerceptionActionSpace[ActType, PredType]:
        return self.__action_space

    @property
    def prediction_target_space(self) -> PredTargetType:
        return self.__prediction_target_space

    @property
    def single_action_space(self) -> ActivePerceptionActionSpace[ActType, PredType]:
        return self.__single_action_space

    @property
    def single_prediction_target_space(self) -> PredTargetType:
        return self.__single_prediction_target_space

    def __getattr__(self, item):
        try:
            return super().__getattribute__(item)
        except AttributeError:
            return getattr(self.env, item)

    def __del__(self):
        # Leave this here to prevent this bug:
        # https://github.com/Farama-Foundation/Gymnasium/issues/1521
        pass


def ensure_active_perception_vector_env(
    env: gym.vector.VectorEnv,
) -> BaseActivePerceptionVectorEnv:
    if isinstance(env, BaseActivePerceptionVectorEnv):
        return env
    try:
        return ActivePerceptionVectorRestoreWrapper(env)
    except NoActivePerceptionEnvError:
        pass
    return PseudoActivePerceptionVectorWrapper(env)
