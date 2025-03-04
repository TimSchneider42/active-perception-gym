#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p "$SCRIPT_DIR/../doc/img"
for env in LightDark-v0 CircleSquare-v0 MNIST-v0 TinyImageNet-v0 CIFAR-v0 Localization2D-v0; do
    python "$SCRIPT_DIR/create_env_gif.py" "$env" "$SCRIPT_DIR/../doc/img/$env.gif"
done