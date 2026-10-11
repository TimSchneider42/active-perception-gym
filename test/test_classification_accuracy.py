"""`accuracy` must mean argmax accuracy: the class the predictor would name.

Pinned on the shipped wrappers rather than on a local copy of the formula, because
the thing these guard against is a refactor of the metric plumbing quietly changing
what the number means. The baseline matters as much as the definition -- a predictor
carrying no information has to read 1/num_classes and nothing else, since that is the
number every result is compared against.
"""

import numpy as np
import pytest
import scipy.special

K = 27


def _scalar_metrics(logits, target):
    from ap_gym.active_classification_env import ActiveClassificationLogWrapper

    w = object.__new__(ActiveClassificationLogWrapper)
    return ActiveClassificationLogWrapper._step_metrics(
        w, {"prediction": np.asarray(logits)}, {"prediction": {"target": target}}
    )


def _vector_metrics(logits, targets):
    from ap_gym.active_classification_env import ActiveClassificationVectorLogWrapper

    class _Bare(ActiveClassificationVectorLogWrapper):
        """`num_envs` is a read-only property on the real class."""

        @property
        def num_envs(self):
            return len(targets)

    w = object.__new__(_Bare)
    return _Bare._step_metrics(
        w,
        {"prediction": np.asarray(logits)},
        {"prediction": {"target": np.asarray(targets)}},
    )


def test_an_uninformative_predictor_reads_as_chance():
    rng = np.random.default_rng(1)
    n = 20000
    logits = rng.normal(0, 0.5, (n, K))
    targets = rng.integers(0, K, n)
    acc = _vector_metrics(logits, targets)["accuracy"].mean()
    assert abs(acc - 1 / K) < 0.01, f"accuracy must sit at chance, got {acc}"


def test_a_confident_predictor_reads_one_and_zero():
    logits = np.full(K, -10.0)
    logits[7] = 10.0
    assert _scalar_metrics(logits, 7)["accuracy"] == 1.0
    assert _scalar_metrics(logits, 8)["accuracy"] == 0.0


def test_only_the_named_class_counts():
    """Favouring the true class is not enough if another class is named."""
    logits = np.zeros(K)
    logits[3] = 0.01  # true class, slightly favoured
    logits[11] = 0.02  # ... but this is the one the predictor would name
    assert _scalar_metrics(logits, 3)["accuracy"] == 0.0
    assert _scalar_metrics(logits, 11)["accuracy"] == 1.0


def test_the_probability_channel_is_untouched():
    """`correct_label_prob` stays the true class's mass, whatever accuracy does."""
    logits = np.zeros(K)
    logits[3] = 0.01
    logits[11] = 0.02
    assert _scalar_metrics(logits, 3)["correct_label_prob"] == pytest.approx(
        scipy.special.softmax(logits)[3]
    )


def test_the_vector_wrapper_agrees_per_env():
    logits = np.zeros((2, K))
    logits[0, 3] = 0.01
    logits[0, 11] = 0.02
    logits[1, 5] = 5.0
    m = _vector_metrics(logits, [3, 5])
    np.testing.assert_array_equal(m["accuracy"], [0.0, 1.0])
    np.testing.assert_allclose(
        m["correct_label_prob"],
        [scipy.special.softmax(logits[0])[3], scipy.special.softmax(logits[1])[5]],
        rtol=1e-6,
    )
