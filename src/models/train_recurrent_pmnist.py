"""This script trains a DNTM on the PMNIST task."""
import click
import torch.nn
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from src.utils import configure_logging, get_str_timestamp, configure_reproducibility, seed_worker
from src.data.perm_seq_mnist import get_dataset

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy
import mlflow
import numpy as np

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--lr", type=float, default=0.00001)
@click.option("--batch_size", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--seed", type=int, default=2147483647)
@click.option("--n_locations", type=int, default=750)
@click.option("--content_size", type=int, default=32)
@click.option("--address_size", type=int, default=8)
@click.option("--permute", type=bool, default=False)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, n_locations, content_size, address_size, lr, batch_size, epochs, permute, seed):
    train_and_test_dntm_smnist(loglevel, run_name, n_locations, content_size, address_size,
                               lr, batch_size, epochs, permute, seed)


def train_and_test_dntm_smnist(loglevel, run_name, n_locations, content_size, address_size,
                               lr, batch_size, epochs, permute, seed):
    run_name = "_".join([train_and_test_dntm_smnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)
    mlflow.set_tracking_uri("file:../logs/mlruns/")
    mlflow.set_experiment(experiment_name="dntm_pmnist")
    writer = SummaryWriter(log_dir=f"../logs/tensorboard/{run_name}")

    device = torch.device("cuda", 0)
    configure_reproducibility(device, seed)
    train, test = get_dataset(permute, seed)

    train.data, train.targets = train.data[:15], train.targets[:15]  # only for debugging

    rng = torch.Generator()
    rng.manual_seed(seed)
    train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=False,
                                   worker_init_fn=seed_worker, num_workers=0, generator=rng)  # reproducibility

    controller_input_size = 1
    controller_output_size = 10
    dntm = build_model(address_size, content_size, controller_input_size, controller_output_size, device,
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
            logging.info(f"Epoch {epoch}")
            output, loss_value, accuracy = training_step(device, dntm, loss_fn, opt, train_data_loader, writer, epoch,
                                                         batch_size)
            writer.add_scalar("Loss/train", loss_value, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)

        logging.info(f"Saving the model")
        torch.save(dntm.state_dict(), f"../models/{run_name}.pth")

        del train_data_loader
        test_data_loader = DataLoader(test, batch_size=batch_size)

        logging.info("Starting testing phase")
        test_step(device, dntm, output, test_data_loader)


def build_model(address_size, content_size, controller_input_size, controller_output_size, device,
                n_locations):
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
    ).to(device)
    return dntm


def training_step(device, model, loss_fn, opt, train_data_loader, writer, epoch, batch_size):
    train_accuracy = Accuracy().to(device)

    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):

        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            writer.add_images(f"Training data batch {batch_i}",
                              mnist_images.reshape(batch_size, 28, 28, 1),
                              dataformats='NHWC')

        logging.debug(f"Resetting the memory")
        model.memory.reset_memory_content()
        model.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        model.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))

        if batch_i == 0:
            writer.add_text(tag="First batch preds vs targets",
                            text_string='pred: ' + ' '.join([str(p.item()) for p in output.argmax(axis=0)]) +
                                        "\n\n target:" + ' '.join([str(t.item()) for t in targets]),
                            global_step=epoch)

        logging.debug(f"Computing loss value")
        loss_value = loss_fn(output.T, targets)

        logging.debug(f"Computing gradients")
        loss_value.backward()
        logging.debug(f"{model.W_output.grad=}")
        logging.debug(f"{model.b_output.grad=}")

        logging.debug(f"Running optimization step")
        opt.step()

        batch_accuracy = train_accuracy(output.argmax(axis=0), targets)
    accuracy_over_batches = train_accuracy.compute()
    train_accuracy.reset()
    return output, loss_value, accuracy_over_batches


def test_step(device, dntm, output, test_data_loader):
    test_accuracy = Accuracy().to(device)

    dntm.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        dntm.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, -1))
            break

        dntm.memory.reset_memory_content()
        batch_accuracy = test_accuracy(output.T, targets)
    test_accuracy = test_accuracy.compute()
    mlflow.log_metric(key="test_accuracy", value=test_accuracy.item())


if __name__ == "__main__":
    click_wrapper()