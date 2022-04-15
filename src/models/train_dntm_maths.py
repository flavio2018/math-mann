"""This script trains a DNTM on the Deepmind dataset of mathematical problems."""
import click
from codetiming import Timer
from humanfriendly import format_timespan

from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform, Sequential, ToTensor

from src.utils import config_run
from src.data.math_dm import MathematicsDataset, yield_chars, collate_fn
from src.data.BucketBatchSampler import BucketBatchSampler


@click.command()
@click.option("--loglevel", type=str, default="INFO")
@click.option("--run_name", type=str, default="")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def click_wrapper(loglevel, run_name):
    train_and_test_dntm_maths(loglevel, run_name)


def train_and_test_dntm_maths(loglevel, run_name):
    run_codename = run_name
    device, rng, run_name, writer = config_run(loglevel, run_name, seed=0)

    problem_name = "arithmetic__mixed"
    print("Building vocabulary...")
    vocabulary = build_vocab_from_iterator(
        yield_chars(f"../data/external/mathematics_dataset-v1.0/train-easy/{problem_name}.txt"))
    ds = MathematicsDataset(problem_name, transform=VocabTransform(vocabulary))

    X, y = ds.numpy()
    bucket_batch_sampler = BucketBatchSampler(X, 16)  # <-- does not store X

    dl = DataLoader(ds, batch_sampler=bucket_batch_sampler, shuffle=False,
                    num_workers=2, drop_last=False, collate_fn=collate_fn)

    for batch, targets in dl:
        print(batch, targets)
        break

    configure_logging(loglevel, run_name)


if __name__ == "__main__":
    click_wrapper()
