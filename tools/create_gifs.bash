#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

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

OUTPUT_DIR="$SCRIPT_DIR/../doc/img"
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
for env in "${ENVS[@]}"; do
  ap-gym-create-env-vid "$env" "${OUTPUT_DIR}/$env.gif" &
done

wait

cd "$OUTPUT_DIR"
ln -s CircleSquare-v0.gif CircleSquare-s28-v0.gif
ln -s MNIST-v0.gif MNIST-train-v0.gif
ln -s CIFAR10-v0.gif CIFAR10-train-v0.gif
ln -s TinyImageNet-v0.gif TinyImageNet-train-v0.gif
ln -s CIFAR10Loc-v0.gif CIFAR10Loc-train-v0.gif
ln -s TinyImageNetLoc-v0.gif TinyImageNetLoc-train-v0.gif
