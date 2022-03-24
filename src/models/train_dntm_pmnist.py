"""This script trains a DNTM on the PMNIST task."""
import click
import torch.nn
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from src.utils import configure_logging, get_str_timestamp
from src.data.perm_seq_mnist import get_dataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy
import mlflow

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--lr", type=float, default=0.00001)
@click.option("--batch_size", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--n_locations", type=int, default=750)
@click.option("--content_size", type=int, default=32)
@click.option("--address_size", type=int, default=8)
@click.option("--permute", type=bool, default=False)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, n_locations, content_size, address_size, lr, batch_size, epochs, permute):
    train_and_test_dntm_smnist(loglevel, run_name, n_locations, content_size, address_size,
                               lr, batch_size, epochs, permute)


def train_and_test_dntm_smnist(loglevel, run_name, n_locations, content_size, address_size,
                               lr, batch_size, epochs, permute):
    run_name = "_".join([train_and_test_dntm_smnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)
    mlflow.set_tracking_uri("file:../logs/mlruns/")
    mlflow.set_experiment(experiment_name="dntm_pmnist")
    writer = SummaryWriter(log_dir=f"../logs/tensorboard/{run_name}")

    train, test = get_dataset(permute)
    train_data_loader = DataLoader(train, batch_size=batch_size)
    device = torch.device("cuda", 0)

    controller_input_size = 1
    controller_output_size = 10
    dntm = build_model(address_size, batch_size, content_size, controller_input_size, controller_output_size, device,
                       n_locations)

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(dntm.parameters(), lr=lr)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "n_locations": n_locations,
                "content_size": content_size,
                "address_size": address_size,
                "controller_input_size": controller_input_size,
                "controller_output_size": controller_output_size
            })

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            output = training_step(batch_size, device, dntm, loss_fn, opt, train_data_loader, writer)
        logging.info(f"Saving the model")
        torch.save(dntm.state_dict(), f"../models/{run_name}.pth")

        del train_data_loader
        test_data_loader = DataLoader(test, batch_size=batch_size)

        logging.info("Starting testing phase")
        test_step(batch_size, device, dntm, output, test_data_loader)


def build_model(address_size, batch_size, content_size, controller_input_size, controller_output_size, device,
                n_locations):
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=content_size,
        address_size=address_size,
        controller_input_size=controller_input_size,
        batch_size=batch_size,
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=n_locations,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size,
        batch_size=batch_size,
    ).to(device)
    return dntm


def test_step(batch_size, device, dntm, output, test_data_loader):
    test_accuracy = Accuracy().to(device)

    dntm.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, batch_size))

        dntm.memory.reset_memory_content()
        batch_accuracy = test_accuracy(output.T, targets)
    test_accuracy = test_accuracy.compute()
    mlflow.log_metric(key="test_accuracy", value=test_accuracy.item())


def training_step(batch_size, device, dntm, loss_fn, opt, train_data_loader, writer):
    train_accuracy = Accuracy().to(device)

    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):
        # logging.debug(f"{torch.cuda.memory_summary()=}")

        logging.info(f"MNIST batch {batch_i}")
        dntm.zero_grad()

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, batch_size))

        logging.debug(f"Computing loss value")
        loss_value = loss_fn(output.T, targets)

        logging.debug(f"Computing gradients")
        loss_value.backward()

        logging.debug(f"Running optimization step")
        opt.step()

        logging.debug(f"Resetting the memory")
        dntm.memory.reset_memory_content()

        writer.add_scalar("Loss/train", loss_value, batch_i)
        batch_accuracy = train_accuracy(output.T, targets)
        writer.add_scalar("Accuracy/train", batch_accuracy, batch_i)
    train_accuracy.reset()
    return output


if __name__ == "__main__":
    click_wrapper()
