#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
export PYTHONPATH="$PYTHONPATH:$SCRIPT_DIR/.."

ENVS=(
  LightDark-v0
  CircleSquare-v0
  CircleSquare-nograd-v0
  CircleSquare-s15-v0
  CircleSquare-s15-nograd-v0
  CircleSquare-s20-v0
  CircleSquare-s20-nograd-v0
  MNIST-v0
  MNIST-test-v0
  TinyImageNet-v0
  TinyImageNet-test-v0
  TinyImageNetLoc-v0
  TinyImageNetLoc-test-v0
  CIFAR10-v0
  CIFAR10-test-v0
  CIFAR10Loc-v0
  CIFAR10Loc-test-v0
  LIDARLocMaze-v0
  LIDARLocMazeStatic-v0
  LIDARLocRooms-v0
  LIDARLocRoomsStatic-v0
)

rm -rf "$SCRIPT_DIR/../doc/img"
mkdir -p "$SCRIPT_DIR/../doc/img"
for env in "${ENVS[@]}"; do
  python "$SCRIPT_DIR/create_env_gif.py" "$env" "$SCRIPT_DIR/../doc/img/$env.gif" &
done

wait
