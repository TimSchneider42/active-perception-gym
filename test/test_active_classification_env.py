import unittest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import jax
except ImportError:
    jax = None

from ap_gym import CrossEntropyLossFn


class TestActiveClassificationEnv(unittest.TestCase):
    def test_cross_entropy_loss_fn(self):
        loss_fn = CrossEntropyLossFn()
        rng = np.random.default_rng(0)
        batch_shape_lengths = rng.integers(0, 5, size=20)
        prediction_shape_lengths = batch_shape_lengths + rng.integers(
            1, 5, size=batch_shape_lengths.shape
        )
        prediction_shapes = [
            tuple(rng.integers(1, 10, size=dim)) for dim in prediction_shape_lengths
        ]
        for batch_shape_length, shape in zip(batch_shape_lengths, prediction_shapes):
            prediction = rng.standard_normal(shape)
            target = rng.integers(0, shape[-1], size=shape[:-1])
            batch_shape = shape[:batch_shape_length]
            expected = loss_fn.numpy(prediction, target, batch_shape)
            with self.subTest("torch", shape=shape):
                if torch is not None:
                    np.testing.assert_allclose(
                        loss_fn.torch(
                            torch.from_numpy(prediction),
                            torch.from_numpy(target),
                            batch_shape,
                        ).numpy(),
                        expected,
                        rtol=1e-4,
                    )
            with self.subTest("jax", shape=shape):
                if jax is not None:
                    np.testing.assert_allclose(
                        np.array(loss_fn.jax(prediction, target, batch_shape)),
                        expected,
                        rtol=1e-4,
                    )
