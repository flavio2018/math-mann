"""This script creates a visualization of the PMNIST DNTM on Tensorboard."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.utils import configure_logging, get_str_timestamp

import torch
from torch.utils.tensorboard import SummaryWriter

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name):
    visualize_dntm_pmnist_model_graph(loglevel, run_name)


def visualize_dntm_pmnist_model_graph(loglevel, run_name):
    run_name = "_".join([visualize_dntm_pmnist_model_graph.__name__, get_str_timestamp(), run_name])
    configure_logging(loglevel, run_name)

    writer = SummaryWriter(log_dir=f'../logs/tensorboard/{run_name}')

    n_locations = 12
    controller_input_size = 1
    controller_output_size = 10

    dntm = DynamicNeuralTuringMachine(
        n_locations=n_locations,
        content_size=8,
        address_size=2,
        controller_hidden_state_size=n_locations,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size
    ).to("cuda")

    mocked_input = torch.ones(size=(1, 1), device="cuda")

    hidden_state, output = dntm(mocked_input)

    hidden_state = hidden_state.detach()

    writer.add_graph(dntm, mocked_input)
    writer.close()


if __name__ == "__main__":
    click_wrapper()
