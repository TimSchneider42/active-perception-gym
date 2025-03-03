from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Tuple, Generic, Callable

import numpy as np
import scipy

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
    ) -> float:
        pass

    def torch(
        self, prediction: Any, target: Any, batch_shape: tuple[int, ...] = ()
    ) -> "torch.Tensor":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def jax(
        self, prediction: Any, target: Any, batch_shape: tuple[int, ...] = ()
    ) -> "jax.Array":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def __call__(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
    ):
        return self.numpy(prediction, target, batch_shape)


class LambdaLossFn(LossFn[PredType, PredTargetType], Generic[PredType, PredTargetType]):
    def __init__(
        self,
        np: Callable[[PredType, PredTargetType, tuple[int, ...]], float],
        torch: Callable[[Any, Any, tuple[int, ...]], "torch.Tensor"] | None = None,
        jax: Callable[[Any, Any, tuple[int, ...]], "jax.Array"] | None = None,
    ):
        self._np = np
        self._torch = torch
        self._jax = jax

    def numpy(
        self,
        prediction: PredType,
        target: PredTargetType,
        batch_shape: tuple[int, ...] = (),
    ) -> float:
        return self._np(prediction, target, batch_shape)

    def torch(
        self, prediction: Any, target: Any, batch_shape: tuple[int, ...] = ()
    ) -> "torch.Tensor":
        if self._torch is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        return self._torch(prediction, target, batch_shape)

    def jax(
        self, prediction: Any, target: Any, batch_shape: tuple[int, ...] = ()
    ) -> "jax.Array":
        if self._jax is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        return self._jax(prediction, target, batch_shape)


class ZeroLossFn(LossFn[Tuple, Tuple]):
    def numpy(
        self, prediction: Tuple, target: Tuple, batch_shape: tuple[int, ...] = ()
    ) -> float:
        return np.zeros(batch_shape, dtype=np.float32)

    def torch(
        self, prediction: Tuple, target: Tuple, batch_shape: tuple[int, ...] = ()
    ) -> "torch.Tensor":
        return torch.zeros(batch_shape)

    def jax(
        self, prediction: Tuple, target: Tuple, batch_shape: tuple[int, ...] = ()
    ) -> "jax.Array":
        return jnp.zeros(batch_shape)


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


class MSELossFn(LossFn[np.ndarray, int | np.ndarray]):
    def numpy(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        batch_shape: tuple[int, ...] = (),
    ) -> float:
        return np.mean((prediction - target) ** 2, axis=-1)

    def torch(
        self,
        prediction: "torch.Tensor",
        target: "torch.Tensor",
        batch_shape: tuple[int, ...] = (),
    ) -> "torch.Tensor":
        return torch.mean((prediction - target) ** 2, dim=-1)

    def jax(
        self,
        prediction: "jax.Array",
        target: "jax.Array",
        batch_shape: tuple[int, ...] = (),
    ) -> "jax.Array":
        return jnp.mean((prediction - target) ** 2, axis=-1)
