"""This script trains a simple recurrent network on the PMNIST task."""
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


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--lr", type=float, default=0.00001)
@click.option("--batch_size", type=int, default=4)
@click.option("--epochs", type=int, default=10)
@click.option("--seed", type=int, default=2147483647)
@click.option("--permute", type=bool, default=False)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, lr, batch_size, epochs, permute, seed):
    train_and_test_recurrent_smnist(loglevel, run_name, lr, batch_size, epochs, permute, seed)


def train_and_test_recurrent_smnist(loglevel, run_name, lr, batch_size, epochs, permute, seed):
    run_name = "_".join([train_and_test_recurrent_smnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)
    mlflow.set_tracking_uri("file:../logs/mlruns/")
    mlflow.set_experiment(experiment_name="recurrent_pmnist")
    writer = SummaryWriter(log_dir=f"../logs/tensorboard/{run_name}")

    device = torch.device("cuda", 0)
    configure_reproducibility(device, seed)
    train, test = get_dataset(permute, seed)

    train.data, train.targets = train.data[:15], train.targets[:15]  # only for debugging

    rng = torch.Generator()
    rng.manual_seed(seed)
    train_data_loader = DataLoader(train, batch_size=batch_size, shuffle=False,
                                   worker_init_fn=seed_worker, num_workers=0, generator=rng)  # reproducibility

    input_size = 1
    output_size = 10
    model = build_model(input_size, output_size, device)

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
                "learning_rate": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "input_size": input_size,
                "output_size": output_size
            })

        torch.autograd.set_detect_anomaly(True)
        for epoch in range(epochs):
            logging.info(f"Epoch {epoch}")
            loss_value, accuracy = training_step(device, model, loss_fn, opt, train_data_loader, writer, epoch, batch_size)
            writer.add_scalar("Loss/train", loss_value, epoch)
            writer.add_scalar("Accuracy/train", accuracy, epoch)


def build_model(input_size, output_size, device):
    return torch.nn.GRU(input_size=input_size,
                        hidden_size=output_size,
                        batch_first=True).to(device)


def training_step(device, model, loss_fn, opt, train_data_loader, writer, epoch, batch_size):
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):

        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            writer.add_images(f"Training data batch {batch_i}",
                              mnist_images.reshape(batch_size, -1, 28, 1),
                              dataformats='NHWC')

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Feeding pixel sequence to model")
        full_output, h_n = model(mnist_images.view(-1, 28*10, 1))
        logging.debug(f"{full_output=}")
        last_output = full_output[:, -1, :].view((-1, 10))  # select only the output for the last timestep and reshape to 2D
        logging.debug(f"{last_output=}")
        log_soft_output = torch.nn.functional.log_softmax(last_output, dim=0)
        logging.debug(f"{log_soft_output=}")

        if batch_i == 0:
            writer.add_text(tag="First batch preds vs targets",
                            text_string='pred: ' + ' '.join([str(p.item()) for p in log_soft_output.argmax(axis=1)]) +
                                        "\n\n target:" + ' '.join([str(t.item()) for t in targets]),
                            global_step=epoch)

        logging.debug(f"Computing loss value")
        logging.debug(f"{targets=}")
        loss_value = loss_fn(log_soft_output, targets)
        # writer.add_scalar("per-batch_loss/train", loss_value, batch_i)
        epoch_loss += loss_value.item()*mnist_images.size(0)

        logging.debug(f"Computing gradients")
        loss_value.backward()
        logging.debug(f"{model._parameters['weight_hh_l0'].grad=}")
        logging.debug(f"{model._parameters['bias_hh_l0'].grad=}")

        logging.debug(f"Running optimization step")
        opt.step()

        batch_accuracy = train_accuracy(log_soft_output.argmax(axis=1), targets)
    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


def test_step(device, model, test_data_loader):
    test_accuracy = Accuracy().to(device)

    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))
            break

        batch_accuracy = test_accuracy(output.T, targets)
    test_accuracy = test_accuracy.compute()
    mlflow.log_metric(key="test_accuracy", value=test_accuracy.item())


if __name__ == "__main__":
    click_wrapper()
