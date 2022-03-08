"""This script trains a DNTM on the PMNIST task."""
import click
import torch.nn
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from src.utils import configure_logging, get_str_timestamp

from toolz.functoolz import compose_left

import numpy as np
from torchvision import datasets
from torchvision.transforms import Lambda
from torch.utils.data import DataLoader


from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name):
    """This wrapper is needed to
    a) import the main method of the script in other scripts, to enable reuse and modularity
    b) allow to access the name of the function in the main method"""
    train_dntm_pmnist(loglevel, run_name)


def train_dntm_pmnist(loglevel, run_name):
    run_name = "_".join([train_dntm_pmnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)

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
    controller_output_size = 10
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=100,
        address_size=20,
        controller_input_size=controller_input_size
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=n_locations,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size
    ).to("cuda")

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(dntm.parameters())

    # TODO handle minibatches?
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(2):
        for i, (mnist_image, target) in enumerate(data_loader):
            logging.info(f"MNIST image {i}")
            dntm.zero_grad()

            logging.debug(f"Moving image to GPU")
            mnist_image, target = mnist_image.to("cuda"), target.to("cuda")

            logging.debug(f"Looping through image pixels")
            for pixel in mnist_image[0]:
                __, output = dntm(pixel.view(1, 1))

            logging.debug(f"Computing loss value")
            loss_value = loss_fn(output.T, target)

            logging.debug(f"Computing gradients")
            loss_value.backward()

            logging.debug(f"Running optimization step")
            opt.step()

            logging.debug(f"Resetting the memory")
            dntm.memory.reset_memory_content()

        logging.info(f"{epoch=}: {loss_value=}")


if __name__ == "__main__":
    click_wrapper()
