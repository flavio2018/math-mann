"""This script trains a DNTM on the PMNIST task."""
import click
import torch.nn
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from src.utils import seed_worker, config_run
from src.data.perm_seq_mnist import get_dataset

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import Accuracy
import mlflow
from src.models.pytorchtools import EarlyStopping

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--seed", type=int, default=2147483647)
@click.option("--ckpt", type=click.Path(exists=True), default=None)
@click.option("--lr", type=float, default=0.00001)
@click.option("--batch_size", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--n_locations", type=int, default=750)
@click.option("--content_size", type=int, default=32)
@click.option("--address_size", type=int, default=8)
@click.option("--permute", type=bool, default=False)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, seed,
                  lr, batch_size, epochs,
                  ckpt, n_locations, content_size, address_size, permute):
    train_and_test_dntm_smnist(loglevel, run_name, seed,
                               lr, batch_size, epochs,
                               ckpt, n_locations, content_size, address_size, permute)


def train_and_test_dntm_smnist(loglevel, run_name, seed,
                               lr, batch_size, epochs,
                               ckpt, n_locations, content_size, address_size, permute):
    run_codename = run_name
    device, rng, run_name, writer = config_run(loglevel, run_name, seed)

    train, test = get_dataset(permute, seed)
    train.data, train.targets = train.data[:600], train.targets[:600]
    test.data, test.targets = test.data[:100], test.targets[:100]

    # obtain training indices that will be used for validation
    valid_size = 0.2
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_data_loader = DataLoader(train,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=train_sampler,
                                   num_workers=0,
                                   generator=rng)  # reproducibility

    valid_data_loader = DataLoader(train,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=valid_sampler,
                                   num_workers=0,
                                   generator=rng)  # reproducibility

    controller_input_size = 1
    controller_output_size = 10
    controller_hidden_state_size = 100
    dntm = build_model(ckpt, address_size, content_size, controller_input_size, controller_output_size,
                       controller_hidden_state_size, device, n_locations)

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(dntm.parameters(), lr=lr)
    early_stopping = EarlyStopping(verbose=True,
                                   path=f"../models/checkpoints/{run_name}.pth",
                                   trace_func=logging.info,
                                   patience=1000)

    with mlflow.start_run(run_name=run_codename):
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

        # training
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch}")

            train_loss, train_accuracy = training_step(device, dntm, loss_fn, opt, train_data_loader, writer, epoch, batch_size)
            valid_loss, valid_accuracy = valid_step(device, dntm, loss_fn, valid_data_loader)

            writer.add_scalars("Loss/training", {'training_set': train_loss,
                                                 'validation_set': valid_loss}, epoch)
            writer.add_scalars("Accuracy/training", {'training_set': train_accuracy,
                                                     'validation_set': valid_accuracy}, epoch)

            early_stopping(valid_loss, dntm)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break

        # testing
        del train_data_loader
        test_data_loader = DataLoader(test, batch_size=batch_size)

        logging.info("Starting testing phase")
        test_step(device, dntm, test_data_loader)


def valid_step(device, dntm, loss_fn, valid_data_loader):
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    dntm.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        dntm.memory.reset_memory_content()
        dntm.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, -1))

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

        batch_accuracy = valid_accuracy(output.argmax(axis=0), targets)
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_data_loader.sampler)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def build_model(ckpt, address_size, content_size, controller_input_size, controller_output_size,
                controller_hidden_state_size, device, n_locations):
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=content_size,
        address_size=address_size,
        controller_input_size=controller_input_size,
        controller_hidden_state_size=controller_hidden_state_size
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=controller_hidden_state_size,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size
    ).to(device)
    if ckpt is not None:
        logging.info(f"Reloading from checkpoint: {ckpt}")
        state_dict = torch.load(ckpt)
        batch_size_ckpt = state_dict['controller_hidden_state'].shape[1]
        dntm.memory.reset_memory_content()
        dntm.reshape_and_reset_hidden_states(batch_size=batch_size_ckpt, device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=batch_size_ckpt, device=device)
        dntm.memory.reshape_and_reset_read_write_weights(shape=state_dict['memory.read_weights'].shape)
        dntm.load_state_dict(state_dict)
    return dntm


def training_step(device, model, loss_fn, opt, train_data_loader, writer, epoch, batch_size):
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):

        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            writer.add_images(f"Training data batch {batch_i}",
                              mnist_images.reshape(batch_size, -1, 28, 1),
                              dataformats='NHWC')

        logging.debug(f"Resetting the memory")
        model.memory.reset_memory_content()
        model.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        model.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)
        model.controller_hidden_state = model.controller_hidden_state.detach()

        if (epoch == 0) and (batch_i == 0):
            mocked_input = torch.ones(size=(1, mnist_images.shape[0]), device="cuda")
            hidden_state, output = model(mocked_input)
            writer.add_graph(model, mocked_input)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))

        if batch_i == 0:
            writer.add_text(tag="First batch preds vs targets",
                            text_string='pred:\t' + ' '.join([str(p.item()) for p in output.argmax(axis=0)]) +
                                        "\n\n target:\t" + ' '.join([str(t.item()) for t in targets]),
                            global_step=epoch)

        logging.debug(f"Computing loss value")
        logging.debug(f"{targets=}")
        loss_value = loss_fn(output.T, targets)
        # writer.add_scalar("per-batch_loss/train", loss_value, batch_i)
        epoch_loss += loss_value.item() * mnist_images.size(0)

        logging.debug(f"Computing gradients")
        loss_value.backward()

        logging.debug(f"Running optimization step")
        opt.step()

        batch_accuracy = train_accuracy(output.argmax(axis=0), targets)
        log_weights_gradient(model, writer, batch_i, epoch)

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


def test_step(device, dntm, test_data_loader):
    test_accuracy = Accuracy().to(device)

    dntm.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        dntm.memory.reset_memory_content()
        dntm.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, -1))

        batch_accuracy = test_accuracy(output.argmax(axis=0), targets)
    test_accuracy = test_accuracy.compute()
    mlflow.log_metric(key="test_accuracy", value=test_accuracy.item())


def log_weights_gradient(dntm, writer, batch_i, epoch):
    for param_name, param in dntm.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"{param_name}_gradient_epoch{epoch}", param.grad, global_step=batch_i)
        else:
            logging.warning(f"{param_name} gradient is None!")


if __name__ == "__main__":
    click_wrapper()
