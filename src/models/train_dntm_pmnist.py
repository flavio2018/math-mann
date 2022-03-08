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


from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main(loglevel):
    configure_logging(loglevel)

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


def configure_logging(loglevel):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logging.basicConfig(
        filename='train_dntm_pmnist.log',
        level=numeric_level,
        format='%(levelname)s:%(message)s',
        filemode="w")


if __name__ == "__main__":
    main()
