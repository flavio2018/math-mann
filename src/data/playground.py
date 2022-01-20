"""Playing around with data"""
import click
from codetiming import Timer
from humanfriendly import format_timespan

import trax.data as td
import trax.layers as tl
from trax.models import Transformer
from trax.optimizers import Adam
from trax.supervised import training

from data_generators import XAndYGenerator

@click.command()
@click.option("--epochs", default=2000, help="Number of epochs.")
@Timer(text=lambda secs: f"Took {format_timespan(secs)}")
def main():
    train_stream = XAndYGenerator(
        "../data/external/mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.txt")
    test_stream = XAndYGenerator(
        "../data/external/mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.txt")

    max_length_x, max_length_y = 53, 12
    preprocessing_pipeline = td.Serial(
        td.tf_inputs.Tokenize(vocab_type='char', n_reserved_ids=1),
        td.inputs.PadToLength(len_map={0: max_length_x, 1: max_length_y}, pad_value={0: 0, 1: 0}),
        td.BucketByLength(boundaries=[ 8, 16, 32, 64],
                          batch_sizes=[64, 32, 16,  8]),
        td.AddLossWeights(),
        td.Log(),
    )

    print("Preprocessing data...")
    preprocessed_train_stream = preprocessing_pipeline(train_stream)
    preprocessed_test_stream = preprocessing_pipeline(test_stream)

    max_vocab_size = 122
    model = Transformer(input_vocab_size=max_vocab_size)
    print(model)

    training_task = training.TrainTask(
        labeled_data=preprocessed_train_stream,
        loss_layer=tl.MaskedSequenceAccuracy(),
        optimizer=Adam(0.01),
        n_steps_per_checkpoint=500,
    )

    eval_task = training.EvalTask(
        labeled_data=preprocessed_test_stream,
        metrics=[tl.MaskedSequenceAccuracy()],
        n_eval_batches=20,
    )

    ckpt_dir = "../models/checkpoints"
    training_loop = training.Loop(model,
                                  training_task,
                                  eval_tasks=[eval_task],
                                  output_dir=ckpt_dir)
    training_loop.run(epochs)


def find_max_vocab_size(preprocessed_stream):
    max_vocab_size = 0
    for x, y in preprocessed_stream:
        max_vocab_size = max(max_vocab_size, max(x), max(y))
    return max_vocab_size + 1  # integer tokens must be in range(max_vocab_size)


def find_max_input_target_lengths(preprocessed_stream):
    max_input_length = 0
    max_target_length = 0
    for x, y in preprocessed_stream:
        max_input_length = max(max_input_length, len(x))
        max_target_length = max(max_target_length, len(y))
    return max_input_length, max_target_length


if __name__ == "__main__":
    main()
