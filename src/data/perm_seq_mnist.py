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
    # _shuffle_digit_array,
)

pmnist_transforms = compose_left(
    np.array,
    _flatten,
    _convert_to_float32,
    # _shuffle_digit_array,
)


def get_perm_mnist():
    return get_dataset(pmnist_transforms)


def get_seq_mnist():
    return get_dataset(smnist_transforms)


def get_dataset(transforms):
    pmnist_train = datasets.MNIST(
        root="../data/external",
        train=True,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )

    pmnist_test = datasets.MNIST(
        root="../data/external",
        train=False,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )
    return pmnist_train, pmnist_test
