import unittest

import numpy as np

from ap_gym import CrossEntropyLossFn, LambdaLossFn, MSELossFn, ZeroLossFn


class TestLossFn(unittest.TestCase):
    def test_deterministic_loss_fns_ignore_rng(self):
        rng = np.random.default_rng(0)
        prediction = rng.standard_normal((4, 3))
        mse = MSELossFn()
        np.testing.assert_allclose(
            mse.numpy(prediction, np.zeros((4, 3)), (4,), rng=rng),
            mse.numpy(prediction, np.zeros((4, 3)), (4,)),
        )
        ce = CrossEntropyLossFn()
        target = np.zeros(4, dtype=np.int_)
        np.testing.assert_allclose(
            ce.numpy(prediction, target, (4,), rng=rng),
            ce.numpy(prediction, target, (4,)),
        )
        np.testing.assert_allclose(
            ZeroLossFn().numpy((), (), (4,), rng=rng), np.zeros(4)
        )

    def test_lambda_loss_fn_without_rng(self):
        loss_fn = LambdaLossFn(
            lambda prediction, target, batch_shape: np.mean(
                (prediction - target) ** 2, axis=-1
            )
        )
        prediction = np.ones((2, 3))
        target = np.zeros((2, 3))
        rng = np.random.default_rng(0)
        np.testing.assert_allclose(
            loss_fn.numpy(prediction, target, (2,), rng=rng), np.ones(2)
        )
        np.testing.assert_allclose(loss_fn(prediction, target, (2,)), np.ones(2))

    def test_lambda_loss_fn_with_rng(self):
        received = []

        def np_loss(prediction, target, batch_shape, rng):
            received.append(rng)
            return np.mean((prediction - target) ** 2, axis=-1)

        loss_fn = LambdaLossFn(np_loss)
        rng = np.random.default_rng(0)
        loss_fn.numpy(np.ones((2, 3)), np.zeros((2, 3)), (2,), rng=rng)
        self.assertIs(received[-1], rng)
        loss_fn.numpy(np.ones((2, 3)), np.zeros((2, 3)), (2,))
        self.assertIsNone(received[-1])

    def test_stochastic_loss_fn_receives_rng_through_wrappers(self):
        received = []

        def np_loss(prediction, target, batch_shape, rng):
            received.append(rng)
            return np.zeros(batch_shape)

        loss_fn = LambdaLossFn(
            np_loss, lower_bound=0.0, blind_guessing_expected_value=1.0
        ).normalized
        rng = np.random.default_rng(0)
        loss_fn.numpy(np.zeros(3), np.zeros(3), (), rng=rng)
        self.assertIs(received[-1], rng)


if __name__ == "__main__":
    unittest.main()
