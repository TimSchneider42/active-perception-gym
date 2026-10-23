"""Episodes an environment flags via the "episode_discard_logs" info key drop out of the episode metrics."""

import numpy as np

from ap_gym.util import update_info_metrics, update_info_metrics_vec


def test_vector_discarded_episode_is_masked():
    metrics = {"accuracy": ([1.0, 0.0], [1.0, 1.0], [0.0])}
    done = np.array([True, True, False])
    info = update_info_metrics_vec(
        {}, metrics, done, discard_logs=np.array([False, True, False])
    )
    scalar = info["stats"]["scalar"]
    np.testing.assert_array_equal(scalar["_final_accuracy"], [True, False, False])
    np.testing.assert_array_equal(scalar["_avg_accuracy"], [True, False, False])
    assert scalar["final_accuracy"][0] == 0.0


def test_vector_default_keeps_every_finished_episode():
    metrics = {"accuracy": ([1.0], [0.0])}
    info = update_info_metrics_vec({}, metrics, np.array([True, True]))
    np.testing.assert_array_equal(
        info["stats"]["scalar"]["_final_accuracy"], [True, True]
    )


def test_single_discarded_episode_is_masked():
    info = update_info_metrics({}, {"accuracy": [0.0, 1.0]}, discard_logs=True)
    assert not info["stats"]["scalar"]["_final_accuracy"]
    assert update_info_metrics({}, {"accuracy": [0.0, 1.0]})["stats"]["scalar"][
        "_final_accuracy"
    ]
