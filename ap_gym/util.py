from __future__ import annotations

from typing import Mapping, Any, Sequence, TypeVar

import numpy as np


def update_dict_recursive(d: dict, u: Mapping):
    return {
        **d,
        **{
            k: update_dict_recursive(d.get(k, {}), v) if isinstance(v, Mapping) else v
            for k, v in u.items()
        },
    }


def _metric_names(metrics: Mapping[str, Any]) -> list[str]:
    """The metric keys of ``metrics``, excluding the ``_<name>`` validity masks."""
    return [n for n in metrics if not (n.startswith("_") and n[1:] in metrics)]


def _valid_values(values, mask) -> np.ndarray:
    """The entries of ``values`` their per-step validity mask marks as valid."""
    v = np.asarray(list(values), dtype=np.float32)
    if mask is None:
        return v
    m = np.asarray(list(mask), dtype=np.bool_)
    return v[m]


def update_info_metrics(
    info: dict[str, Any], metrics: dict[str, Sequence[float] | np.ndarray]
) -> dict[str, Any]:
    scalar: dict[str, Any] = {}
    vector: dict[str, Any] = {}
    for n in _metric_names(metrics):
        v = _valid_values(metrics[n], metrics.get(f"_{n}"))
        valid = len(v) > 0
        scalar[f"avg_{n}"] = float(v.mean()) if valid else np.nan
        scalar[f"_avg_{n}"] = valid
        scalar[f"final_{n}"] = float(v[-1]) if valid else np.nan
        scalar[f"_final_{n}"] = valid
        # Need to use a list and not a numpy array here as otherwise SyncVectorEnv will
        # try to stack the arrays, which throws if they are of different lengths.
        vector[n] = list(v)
        vector[f"_{n}"] = valid
    return update_dict_recursive(info, {"stats": {"scalar": scalar, "vector": vector}})


def update_info_metrics_vec(
    info: dict[str, Any],
    metrics: dict[str, Sequence[Sequence[float] | np.ndarray]],
    done: np.ndarray,
) -> dict[str, Any]:
    scalar: dict[str, Any] = {}
    vector: dict[str, Any] = {}
    for n in _metric_names(metrics):
        mask = metrics.get(f"_{n}")
        per_env = [
            _valid_values(e, None if mask is None else mask[i]) if t else np.empty(0)
            for i, (e, t) in enumerate(zip(metrics[n], done))
        ]
        # A metric can be invalid for a whole episode -- every step of it undefined, or
        # the episode having no steps at all -- so validity is per environment and not
        # simply `done`.
        valid = np.array([len(v) > 0 for v in per_env], dtype=np.bool_)
        scalar[f"avg_{n}"] = np.array(
            [v.mean() if len(v) else np.nan for v in per_env], dtype=np.float32
        )
        scalar[f"_avg_{n}"] = valid
        scalar[f"final_{n}"] = np.array(
            [v[-1] if len(v) else np.nan for v in per_env], dtype=np.float32
        )
        scalar[f"_final_{n}"] = valid
        # The None trick is to ensure that numpy does not try to stack the lists if they
        # happen to have the same length.
        vector[n] = np.array([list(v) for v in per_env] + [None], dtype=object)[:-1]
        vector[f"_{n}"] = valid
    return update_dict_recursive(
        info,
        {
            "stats": {
                "scalar": scalar,
                "_scalar": done,
                "vector": vector,
                "_vector": done,
            }
        },
    )


T = TypeVar("T")


def idoc(obj: T, doc: Any) -> T:
    obj.__idoc__ = doc
    return obj


def project_sphere(x: np.ndarray, radius: float = 1.0) -> np.ndarray:
    magnitude = np.linalg.norm(x, axis=-1, keepdims=True)
    direction = x / np.maximum(magnitude, radius)
    return np.where(magnitude > radius, direction * radius, x)
