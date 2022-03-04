"""This script trains a DNTM on the PMNIST task."""
import click
import torch.nn
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from toolz.functoolz import compose_left

import numpy as np
from torchvision import datasets
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine, DynamicNeuralTuringMachineMemory


@click.command()
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    def _convert_to_float32(x: np.array):
        return x.astype(np.float32)

    def _flatten(x: np.array):
        return x.flatten()

    def _shuffle_digit_array(x):
        rng = np.random.default_rng()
        rng.shuffle(x)
        return x

    transforms = compose_left(
         np.array,
         _flatten,
         _convert_to_float32,
         _shuffle_digit_array,
    )

    pmnist = datasets.MNIST(
        root="../data/external",
        train=True,
        download=True,
        transform=Lambda(lambda x: transforms(x)),
    )

    data_loader = DataLoader(pmnist, batch_size=1)

    n_locations = 150
    controller_input_size = 1
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=100,
        address_size=20,
        controller_input_size=controller_input_size
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=n_locations,
        controller_input_size=controller_input_size
    )

    for epoch in range(1):
        for sample, target in data_loader:
            for pixel in sample[0]:
                dntm(pixel.view(1, 1))
            dntm.memory.reset_memory_content()
            break


if __name__ == "__main__":
    main()
