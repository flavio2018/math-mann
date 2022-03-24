"""This file contains utility functions to create the Sequential or Permutated MNIST problem."""
from toolz.functoolz import compose_left

import numpy as np
from torchvision import datasets
from torchvision.transforms import Lambda


def _convert_to_float32(x: np.array):
    return x.astype(np.float32)


def _flatten(x: np.array):
    return x.flatten()


def _shuffle_digit_array(x):
    rng = np.random.default_rng(seed=123456)
    # ^ the permutation should be the same for all digits
    rng.shuffle(x)
    return x


smnist_transforms = compose_left(
    np.array,
    _flatten,
    _convert_to_float32,
)

pmnist_transforms = compose_left(
    np.array,
    _flatten,
    _convert_to_float32,
    _shuffle_digit_array,
)


def get_dataset(permute):
    if permute:
        transforms = pmnist_transforms
    else:
        transforms = smnist_transforms

    train = datasets.MNIST(
        root="../data/external",
        train=True,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )

    test = datasets.MNIST(
        root="../data/external",
        train=False,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )
    return train, test
