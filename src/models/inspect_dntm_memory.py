"""This script implements a visualization of the dynamic of use of the DNTM memory on the SMNIST task."""
import click
from codetiming import Timer
from humanfriendly import format_timespan
import logging

from src.utils import configure_logging, get_str_timestamp
from src.models.train_dntm_pmnist import build_model

import numpy as np
import matplotlib.pyplot as plt

import torch


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@click.option("--ckpt", type=str, default=None)
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name, ckpt):
    inspect_dntm_memory(loglevel, run_name, ckpt)


def inspect_dntm_memory(loglevel, run_name, ckpt):
    run_name = "_".join([inspect_dntm_memory.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)

    logging.info(f"Loading model from ckpt: {ckpt}")
    with torch.no_grad():
        n_locations = 785
        dntm = build_model(ckpt, address_size=8, content_size=16, controller_input_size=1, controller_output_size=10,
                           device=torch.device("cpu"), n_locations=n_locations)
        step=100
        for loc in range(0, 700, step):
            hinton(dntm.memory._full_memory_view()[loc:loc+step, :].T)
            plt.savefig(f"../reports/figures/{run_name}_loc{loc+step}.png", dpi=300)
        hinton(dntm.memory._full_memory_view()[loc+step:, :].T)
        plt.savefig(f"../reports/figures/{run_name}_last.png", dpi=300)


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log2(np.abs(matrix).max()))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(abs(w) / max_weight)
        rect = plt.Rectangle((x - size / 2, y - size / 2), size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


if __name__ == "__main__":
    click_wrapper()
