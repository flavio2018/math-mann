"""Describe the purpose of this script here..."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from src.utils import configure_logging, get_str_timestamp


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name):
    """This wrapper is needed to
    a) import the main method of the script in other scripts, to enable reuse and modularity
    b) allow to access the name of the function in the main method"""
    changeme(loglevel, run_name)


def changeme(loglevel, run_name):
    """The logic of the script goes here..."""
    run_name = "_".join([train_dntm_pmnist.__name__, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)


if __name__ == "__main__":
    click_wrapper()
