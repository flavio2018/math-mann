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
import mlflow


from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--lr", type=float, default=0.00001)
@click.option("--n_locations", type=int, default=750)
@click.option("--content_size", type=int, default=32)
@click.option("--address_size", type=int, default=8)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, n_locations, content_size, address_size, lr):
    """This wrapper is needed to
    a) import the main method of the script in other scripts, to enable reuse and modularity
    b) allow to access the name of the function in the main method"""
    train_dntm_pmnist(loglevel, run_name, n_locations, content_size, address_size, lr)


def train_dntm_pmnist(loglevel, run_name, n_locations, content_size, address_size, lr):
    run_name = "_".join([train_dntm_pmnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)
    mlflow.set_tracking_uri("file:../logs/mlruns/")
    mlflow.set_experiment(experiment_name="dntm_pmnist")

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

    controller_input_size = 1
    controller_output_size = 10
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=content_size,
        address_size=address_size,
        controller_input_size=controller_input_size
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=n_locations,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size
    ).to("cuda")

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(dntm.parameters(), lr=lr)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("learning_rate", lr)
        mlflow.log_params({
                "n_locations": n_locations,
                "content_size": content_size,
                "address_size": address_size,
                "controller_input_size": controller_input_size,
                "controller_output_size": controller_output_size
            })

        # TODO handle minibatches?
        torch.autograd.set_detect_anomaly(True)
        for epoch in range(2):
            for mnist_i, (mnist_images, targets) in enumerate(data_loader):
                logging.debug(f"{torch.cuda.memory_summary()=}")

                logging.info(f"MNIST image {mnist_i}")
                dntm.zero_grad()

                logging.debug(f"Moving image to GPU")
                mnist_image, target = mnist_image.to("cuda"), target.to("cuda")

                logging.debug(f"Looping through image pixels")
                for pixel_i, pixel in enumerate(mnist_image[0]):
                    logging.debug(f"Pixel {pixel_i}")
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
