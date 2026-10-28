# Baseline Data

This directory contains the learning curves of the baselines evaluated on the environments of this package.
They are the data behind the plots on the [project page](https://timschneider42.github.io/tactile-mnist/).
Every method was trained for 5M environment steps on 5 seeds per environment.

## Layout

```
baseline-data/<environment>/<method>/<metric>.csv
baseline-data/<environment>/<method>/<metric>_detailed.csv
```

`<environment>` is the environment name without version suffix.
`<method>` is one of
- `crossq`: [CrossQ](https://arxiv.org/abs/1902.05605),
- `sac`: [Soft Actor-Critic](https://arxiv.org/abs/1801.01290),
- `ppo`: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347),
- `ham`: the hard attention model of [Mnih et al. (2014)](https://arxiv.org/abs/1406.6247),
- `rand_act`: random actions; only the predictor is trained.

On `LIDARLocRooms` and `LIDARLocMaze`, only `crossq`, `sac` and `rand_act` were run.

## File formats

`<metric>.csv` holds the curve aggregated over seeds:

```
x,y_mean,y_std
```

`<metric>_detailed.csv` holds the curve of each seed:

```
x,seed_0,seed_1,seed_2,seed_3,seed_4
```

`x` is the number of training environment steps (summed over all parallel environments).
Each seed's curve is smoothed with a Gaussian filter (standard deviation of 3% of the training length) and resampled at 50 evenly spaced points.
`y_mean` and `y_std` are the mean and (population) standard deviation over seeds at each point.
`nan` in a `_detailed.csv` marks points a run did not reach.

## Metrics

Metrics prefixed with `train_` are measured on the training episodes.
Metrics prefixed with `eval_` are measured on separate evaluation episodes of the test split (`<environment>-test-v0`), for environments that have one; these environments are trained on `<environment>-train-v0`.
All values are per-episode quantities averaged over the episodes completed around each point.

| Metric               | Meaning                                                                                                           |
|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `*_ret`              | Episodic return.                                                                                                  |
| `*_loss`             | Prediction loss summed over the episode.                                                                          |
| `*_mean_loss`        | Prediction loss averaged over the steps of the episode.                                                           |
| `*_avg`              | Classification: accuracy averaged over the steps of the episode. Regression: Euclidean distance averaged likewise. |
| `*_final`            | Classification: accuracy at the last step of the episode. Regression: Euclidean distance at the last step.         |

## Environments

| Directory             | Task           | Evaluation split |
|-----------------------|----------------|------------------|
| `CircleSquare`        | Classification | no               |
| `MNIST`               | Classification | yes              |
| `CIFAR10`             | Classification | yes              |
| `TinyImageNet`        | Classification | yes              |
| `CIFAR10Search`       | Regression     | yes              |
| `TinyImageNetSearch`  | Regression     | yes              |
| `LightDark`           | Regression     | no               |
| `LIDARLocRooms`       | Regression     | no               |
| `LIDARLocMaze`        | Regression     | no               |
| `LIDARLocRoomsStatic` | Regression     | no               |
| `LIDARLocMazeStatic`  | Regression     | no               |

`ham` crashed early on two `LightDark` seeds, so their curves end early and `LightDark/ham/*.csv` aggregates the remaining seeds there.
