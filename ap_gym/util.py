from typing import Mapping, Any, Sequence

import numpy as np


def update_dict_recursive(d: dict, u: Mapping):
    return {
        **d,
        **{
            k: update_dict_recursive(d.get(k, {}), v) if isinstance(v, Mapping) else v
            for k, v in u.items()
        },
    }


def update_info_metrics(
    info: dict[str, Any], metrics: dict[str, Sequence[float] | np.ndarray]
) -> dict[str, Any]:
    return update_dict_recursive(
        info,
        {
            "stats": {
                "scalar": {
                    **{f"avg_{n}": float(np.mean(v)) for n, v in metrics.items()},
                    **{f"final_{n}": float(v[-1]) for n, v in metrics.items()},
                },
                "vector": {
                    n: np.asarray(v, dtype=np.float32) for n, v in metrics.items()
                },
            }
        },
    )


def update_info_metrics_vec(
    info: dict[str, Any],
    metrics: dict[str, Sequence[Sequence[float] | np.ndarray]],
    terminated: np.ndarray,
) -> dict[str, Any]:
    return update_dict_recursive(
        info,
        {
            "stats": {
                "scalar": {
                    **{
                        f"final_{n}": np.array(
                            [e[-1] if t else np.nan for t, e in zip(terminated, v)],
                            dtype=np.float32,
                        )
                        for n, v in metrics.items()
                    },
                    **{f"_final_{n}": terminated for n in metrics.keys()},
                    **{
                        f"avg_{n}": np.array(
                            [
                                np.mean(e) if t else np.nan
                                for t, e in zip(terminated, v)
                            ],
                            dtype=np.float32,
                        )
                        for n, v in metrics.items()
                    },
                    **{f"_avg_{n}": terminated for n in metrics.keys()},
                },
                "_scalar": terminated,
                "vector": {
                    **{
                        n: tuple(
                            np.asarray(e, dtype=np.float32)
                            if t
                            else np.array((), dtype=np.float32)
                            for e, t in zip(v, terminated)
                        )
                        for n, v in metrics.items()
                    },
                    **{f"_{n}": terminated for n in metrics.keys()},
                },
                "_vector": terminated,
            }
        },
    )
