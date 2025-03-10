#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p "$SCRIPT_DIR/../doc/img"
for env in LightDark-v0 CircleSquare-v0 CircleSquare-nograd-v0 CircleSquare-s15-v0 CircleSquare-s15-nograd-v0 CircleSquare-s20-v0 CircleSquare-s20-nograd-v0 MNIST-v0 TinyImageNet-v0 CIFAR10-v0 LIDARLocMaze-v0 LIDARLocMazeStatic-v0 TinyImageNetLoc-v0 CIFAR10Loc-v0; do
    python "$SCRIPT_DIR/create_env_gif.py" "$env" "$SCRIPT_DIR/../doc/img/$env.gif" &
done

wait
