from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Generic, Callable

import numpy as np
import scipy
import gymnasium as gym

from .types import PredType, PredTargetType

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


class LossFn(Generic[PredType, PredTargetType], ABC):
    @abstractmethod
    def numpy(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        pass

    def torch(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def jax(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def __call__(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ):
        return self.numpy(prediction, target, batch_shape, rng)

    @property
    def lower_bound(self) -> float:
        """Returns a lower bound of this loss function given the target."""
        return self._lower_bound()

    def _lower_bound(self):
        return -np.inf

    @property
    def blind_guessing_expected_value(self) -> float | None:
        return self._blind_guessing_expected_value()

    def _blind_guessing_expected_value(self) -> float | None:
        """Returns the expected value of this loss function when the prediction is a blind guess."""
        return None

    @property
    def normalized(self):
        upper_bound = self.blind_guessing_expected_value
        if upper_bound is None:
            raise ValueError(
                "Cannot normalize loss function without blind guessing expected value."
            )
        lower_bound = self.lower_bound
        if upper_bound <= lower_bound:
            raise ValueError(
                "Cannot normalize loss function when blind guessing expected value is not greater than lower bound."
            )
        scale = 1 / (upper_bound - lower_bound)
        offset = -lower_bound * scale
        return LossFnAffineTransformation(self, scale, offset)


class LossFnAffineTransformation(
    LossFn[PredType, PredTargetType], Generic[PredType, PredTargetType]
):
    def __init__(
        self,
        inner_loss_fn: LossFn[PredType, PredTargetType],
        scale: float,
        offset: float,
    ):
        super().__init__()
        self.__inner_loss_fn = inner_loss_fn
        self.__scale = scale
        self.__offset = offset

    def numpy(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return (
            self.__inner_loss_fn.numpy(
                prediction, target, batch_shape=batch_shape, rng=rng
            )
            * self.__scale
            + self.__offset
        )

    def torch(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        return (
            self.__inner_loss_fn.torch(
                prediction, target, batch_shape=batch_shape, rng=rng
            )
            * self.__scale
            + self.__offset
        )

    def jax(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        return (
            self.__inner_loss_fn.jax(
                prediction, target, batch_shape=batch_shape, rng=rng
            )
            * self.__scale
            + self.__offset
        )

    def _lower_bound(self) -> float:
        return self.__inner_loss_fn.lower_bound * self.__scale + self.__offset

    def _blind_guessing_expected_value(self) -> float | None:
        inner_value = self.__inner_loss_fn.blind_guessing_expected_value
        if inner_value is None:
            return None
        return inner_value * self.__scale + self.__offset


def _accepts_rng(fn: Callable) -> bool:
    """Checks whether the given callable can be invoked with an additional rng argument."""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    if any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        or p.name == "rng"
        for p in parameters
    ):
        return True
    positional = [
        p
        for p in parameters
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 4


class LambdaLossFn(LossFn[PredType, PredTargetType], Generic[PredType, PredTargetType]):
    def __init__(
        self,
        np: (
            Callable[[PredType, PredTargetType, tuple[int, ...]], np.ndarray]
            | Callable[
                [
                    PredType,
                    PredTargetType,
                    tuple[int, ...],
                    "np.random.Generator | None",
                ],
                np.ndarray,
            ]
        ),
        torch: (
            Callable[[Any, Any, tuple[int, ...]], "torch.Tensor"]
            | Callable[
                [Any, Any, tuple[int, ...], "torch.Generator | None"], "torch.Tensor"
            ]
            | None
        ) = None,
        jax: (
            Callable[[Any, Any, tuple[int, ...]], "jax.Array"]
            | Callable[[Any, Any, tuple[int, ...], "jax.Array | None"], "jax.Array"]
            | None
        ) = None,
        lower_bound: float = -np.inf,
        blind_guessing_expected_value: float | None = None,
    ):
        self.__np = np
        self.__torch = torch
        self.__jax = jax
        self.__np_accepts_rng = _accepts_rng(np)
        self.__torch_accepts_rng = torch is not None and _accepts_rng(torch)
        self.__jax_accepts_rng = jax is not None and _accepts_rng(jax)
        self.__lower_bound = lower_bound
        self.__blind_guessing_expected_value = blind_guessing_expected_value

    def numpy(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        if self.__np_accepts_rng:
            return self.__np(prediction, target, batch_shape, rng)
        return self.__np(prediction, target, batch_shape)

    def torch(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        if self.__torch is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        if self.__torch_accepts_rng:
            return self.__torch(prediction, target, batch_shape, rng)
        return self.__torch(prediction, target, batch_shape)

    def jax(
        self,
        prediction: Any,
        target: Any,
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        if self.__jax is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        if self.__jax_accepts_rng:
            return self.__jax(prediction, target, batch_shape, rng)
        return self.__jax(prediction, target, batch_shape)

    def _lower_bound(self) -> float:
        return self.__lower_bound

    def _blind_guessing_expected_value(self) -> float | None:
        return self.__blind_guessing_expected_value


class ZeroLossFn(LossFn[tuple, tuple]):
    def numpy(
        self,
        prediction: tuple,
        target: tuple,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return np.zeros(batch_shape, dtype=np.float32)

    def torch(
        self,
        prediction: tuple,
        target: tuple,
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        return torch.zeros(batch_shape)

    def jax(
        self,
        prediction: tuple,
        target: tuple,
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        return jnp.zeros(batch_shape)

    def _lower_bound(self) -> float:
        return 0.0

    def _blind_guessing_expected_value(self) -> float:
        return 0.0


class CrossEntropyLossFn(LossFn[np.ndarray, np.ndarray]):
    def __init__(self, num_classes: int | None = None):
        super().__init__()
        self.__num_classes = num_classes

    def numpy(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return -np.take_along_axis(
            scipy.special.log_softmax(prediction, axis=-1), target[..., None], axis=-1
        )[..., 0]

    def torch(
        self,
        prediction: "torch.Tensor",
        target: "torch.Tensor",
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        return -torch.take_along_dim(
            torch.nn.functional.log_softmax(prediction, dim=-1),
            target[..., None],
            dim=-1,
        )[..., 0]

    def jax(
        self,
        prediction: "jax.Array",
        target: "jax.Array",
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        return -jnp.take_along_axis(
            jax.nn.log_softmax(prediction), target[..., None], axis=-1
        )[..., 0]

    def _lower_bound(self):
        return 0.0

    def _blind_guessing_expected_value(self) -> float | None:
        if self.__num_classes is None:
            return None
        return np.log(self.__num_classes)


class MSELossFn(LossFn[np.ndarray, np.ndarray]):
    def __init__(self, target_std: float | np.ndarray | None = None):
        super().__init__()
        if target_std is None:
            self.__blind_guessing_expected_value = None
        else:
            self.__blind_guessing_expected_value = float(np.mean(target_std**2))

    def numpy(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        return np.mean((prediction - target) ** 2, axis=-1)

    def torch(
        self,
        prediction: "torch.Tensor",
        target: "torch.Tensor",
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        return torch.mean((prediction - target) ** 2, dim=-1)

    def jax(
        self,
        prediction: "jax.Array",
        target: "jax.Array",
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        return jnp.mean((prediction - target) ** 2, axis=-1)

    def _lower_bound(self):
        return 0.0

    def _blind_guessing_expected_value(self) -> float | None:
        return self.__blind_guessing_expected_value


class WeightedLossFn(
    LossFn[PredType, dict[str, PredTargetType | np.ndarray]],
    Generic[PredType, PredTargetType],
):
    def __init__(
        self,
        inner_loss_fn: LossFn[PredType, PredTargetType],
        min_weight: float = 0.0,
        average_weight: float | None = None,
    ):
        super().__init__()
        self.__inner_loss_fn = inner_loss_fn
        self.__average_weight = average_weight
        self.__min_weight = min_weight

    def numpy(
        self,
        prediction: PredType,
        target: dict[str, PredTargetType | np.ndarray],
        batch_shape: tuple[int, ...] = (),
        rng: np.random.Generator | None = None,
    ) -> float:
        return (
            self.__inner_loss_fn.numpy(prediction, target["target"], batch_shape, rng)
            * target["weight"]
        )

    def torch(
        self,
        prediction: "torch.Tensor",
        target: "dict[str, PredTargetType | torch.Tensor]",
        batch_shape: tuple[int, ...] = (),
        rng: "torch.Generator | None" = None,
    ) -> "torch.Tensor":
        return (
            self.__inner_loss_fn.torch(prediction, target["target"], batch_shape, rng)
            * target["weight"]
        )

    def jax(
        self,
        prediction: "jax.Array",
        target: "dict[str, PredTargetType | jax.Array]",
        batch_shape: tuple[int, ...] = (),
        rng: "jax.Array | None" = None,
    ) -> "jax.Array":
        return (
            self.__inner_loss_fn.jax(prediction, target["target"], batch_shape, rng)
            * target["weight"]
        )

    def _lower_bound(self):
        return self.__min_weight * self.__inner_loss_fn.lower_bound

    def _blind_guessing_expected_value(self) -> float | None:
        inner_value = self.__inner_loss_fn.blind_guessing_expected_value
        if inner_value is None:
            return None
        if self.__average_weight is None:
            return None
        return self.__average_weight * inner_value
