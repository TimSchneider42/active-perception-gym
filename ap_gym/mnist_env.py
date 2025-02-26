from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import requests
from mnist import MNIST

from .image_classification_vector_env import ImageClassificationVectorEnv
from .vector_to_single_wrapper import ActivePerceptionVectorToSingleWrapper

logger = logging.getLogger(__name__)


class MNISTVectorEnv(ImageClassificationVectorEnv):
    def __init__(
        self,
        num_envs: int,
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        max_episode_steps: int = 32,
        max_step_length: float = 0.2,
        interpolation_method: str = "linear",
    ):
        data_path = Path.home() / ".local" / "share" / "mnist-data"
        mnist = MNIST(str(data_path), return_type="numpy")
        try:
            loaded = mnist.load_training()
        except FileNotFoundError:
            logger.info("Downloading MNIST dataset...")
            base_url = "https://huggingface.co/spaces/chrisjay/mnist-adversarial/resolve/603879aa/files/MNIST/raw"
            files = [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ]
            data_path.mkdir(parents=True, exist_ok=True)
            for file in files:
                filename = data_path / file
                logger.info(f"Downloading {file}...")
                with filename.open("wb") as f:
                    f.write(requests.get(f"{base_url}/{file}").content)
                logger.info(f"Extracting {file}...")
                subprocess.run(["gunzip", str(data_path / file)])
            loaded = mnist.load_training()
            logger.info("Done.")
        self.__images, self.__labels = loaded
        self.__images = self.__images.reshape((-1, 28, 28)) / 255
        super().__init__(
            num_envs,
            len(self.__images),
            10,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            interpolation_method=interpolation_method,
        )

    def _load_image(self, idx: int) -> tuple[np.ndarray, int]:
        return self.__images[idx][..., None], self.__labels[idx]


def MNISTEnv(
    render_mode: Literal["rgb_array", "human"] = "rgb_array",
    max_episode_steps: int = 32,
    max_step_length: float = 1.0,
    interpolation_method: str = "linear",
):
    return ActivePerceptionVectorToSingleWrapper(
        MNISTVectorEnv(
            1,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            max_step_length=max_step_length,
            interpolation_method=interpolation_method,
        )
    )
