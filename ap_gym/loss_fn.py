from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, TYPE_CHECKING, Generic, Callable

import numpy as np

from .types import PredType, PredTargetType

if TYPE_CHECKING:
    import torch
    import jax

try:
    import jax.numpy as jnp
except ImportError:
    jnp = None


class LossFn(Generic[PredType, PredTargetType], ABC):
    @abstractmethod
    def numpy(
        self, prediction: PredType, target: PredTargetType, batch_shape: Tuple[int, ...] = ()
    ) -> float:
        pass

    def torch(
        self, prediction: Any, target: Any, batch_shape: Tuple[int, ...] = ()
    ) -> "torch.Tensor":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def jax(
        self, prediction: Any, target: Any, batch_shape: Tuple[int, ...] = ()
    ) -> "jax.Array":
        raise NotImplementedError("Loss function is not implemented for torch.")

    def __call__(
        self, prediction: PredType, target: PredTargetType, batch_shape: Tuple[int, ...] = ()
    ):
        return self.numpy(prediction, target, batch_shape)


class LambdaLossFn(LossFn[PredType, PredTargetType], Generic[PredType, PredTargetType]):
    def __init__(
        self,
        np: Callable[[PredType, PredTargetType, Tuple[int, ...]], float],
        torch: Optional[Callable[[Any, Any, Tuple[int, ...]], "torch.Tensor"]] = None,
        jax: Optional[Callable[[Any, Any, Tuple[int, ...]], "jax.Array"]] = None,
    ):
        self._np = np
        self._torch = torch
        self._jax = jax

    def numpy(
        self, prediction: PredType, target: PredTargetType, batch_shape: Tuple[int, ...] = ()
    ) -> float:
        return self._np(prediction, target, batch_shape)

    def torch(
        self, prediction: Any, target: Any, batch_shape: Tuple[int, ...] = ()
    ) -> "torch.Tensor":
        if self._torch is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        return self._torch(prediction, target, batch_shape)

    def jax(
        self, prediction: Any, target: Any, batch_shape: Tuple[int, ...] = ()
    ) -> "jax.Array":
        if self._jax is None:
            raise NotImplementedError("Loss function is not implemented for torch.")
        return self._jax(prediction, target, batch_shape)


class ZeroLossFn(LossFn[Tuple, Tuple]):
    def numpy(
        self, prediction: Tuple, target: Tuple, batch_shape: Tuple[int, ...] = ()
    ) -> float:
        return np.zeros(batch_shape, dtype=np.float32)

    def torch(
        self, prediction: Tuple, target: Tuple, batch_shape: Tuple[int, ...] = ()
    ) -> "torch.Tensor":
        return torch.zeros(batch_shape)

    def jax(
        self, prediction: Tuple, target: Tuple, batch_shape: Tuple[int, ...] = ()
    ) -> "jax.Array":
        return jnp.zeros(batch_shape)
