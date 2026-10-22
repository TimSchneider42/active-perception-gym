import unittest

import numpy as np

import scipy.optimize

from ap_gym import (
    CrossEntropyLossFn,
    HuberBlindGuessStatistics,
    HuberLossFn,
    LambdaLossFn,
    MSELossFn,
    ZeroLossFn,
)


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

    def test_mse_from_data(self):
        targets = np.random.default_rng(0).lognormal(size=(1000, 3))
        loss_fn = MSELossFn.from_data(targets)
        guess = np.broadcast_to(targets.mean(axis=0), targets.shape)
        np.testing.assert_allclose(
            loss_fn.normalized.numpy(guess, targets, (len(targets),)).mean(), 1.0
        )
        # Scalar targets may come as a 1-D array
        self.assertAlmostEqual(
            MSELossFn.from_data(targets[:, 0]).blind_guessing_expected_value,
            np.var(targets[:, 0]),
        )

    def test_huber_from_data_uses_the_best_blind_guess(self):
        # Skewed targets, where the best blind guess under the Huber loss is not the mean
        targets = np.random.default_rng(0).lognormal(size=(2000, 2)) * [1.0, 3.0]
        delta = 1.0
        loss_fn = HuberLossFn.from_data(delta, targets)
        raw = HuberLossFn(delta)
        for i in range(targets.shape[1]):
            column = targets[:, i : i + 1]

            def mean_loss(c):
                return raw.numpy(np.full_like(column, c), column, (len(column),)).mean()

            best = scipy.optimize.minimize_scalar(
                mean_loss, bounds=(column.min(), column.max()), method="bounded"
            )
            self.assertLess(best.fun, mean_loss(column.mean()))
            stats = loss_fn.blind_guess_statistics
            column_stats = HuberBlindGuessStatistics(
                delta,
                stats.fraction_above_delta[i],
                stats.mean_abs_error_above_delta[i],
                stats.mean_squared_error_below_delta[i],
            )
            self.assertAlmostEqual(column_stats.expected_loss, best.fun, places=6)
        # The best blind guess scores exactly 1 under the normalized loss
        self.assertAlmostEqual(loss_fn.normalized.blind_guessing_expected_value, 1.0)

    def test_huber_statistics_must_match_delta(self):
        stats = HuberLossFn.from_data(1.0, np.arange(10.0)).blind_guess_statistics
        with self.assertRaises(ValueError):
            HuberLossFn(2.0, stats)
        self.assertIsNone(HuberLossFn(1.0).blind_guessing_expected_value)


if __name__ == "__main__":
    unittest.main()
