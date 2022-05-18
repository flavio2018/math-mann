import numpy as np
import torch
from torch.utils.data import Dataset
import hydra
import os

from src.data.BucketBatchSampler import BucketBatchSampler
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.transforms import VocabTransform


class MathematicsDataset(Dataset):
    """Dataset of mathematical problems first defined by Google Deepmind."""
    def __init__(self, cfg, transform=None):
        self.path = os.path.join(hydra.utils.get_original_cwd(),
                                 f"../data/external/mathematics_dataset-v1.0/train-{cfg.data.problem_difficulty}/{cfg.data.problem_name}.txt")
        with open(self.path) as f:
            dataset = f.readlines()
        self.samples = dataset[::2]
        self.samples = [list(s) for s in self.samples]  # make sequence of chars
        self.targets = dataset[1::2]
        self.targets = [list(t) for t in self.targets]
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        samples = self.samples[idx]
        targets = self.targets[idx]

        if self.transform:
            samples = self.transform(samples)
            targets = self.transform(targets)

        return samples, targets


def to_numpy(ds: Dataset):
    return (np.array([np.array(seq) for seq in ds[:][0]], dtype=object),
            np.array([np.array(target_seq) for target_seq in ds[:][1]], dtype=object))


def to_numpy_subset(subset_ds: torch.utils.data.dataset.Subset):
    """The access to Subsets is different from Datasets. They internally store the original dataset
    and the indices of the subset, then they access the original dataset using them.
    In this function, iterating over the indices of the subset we access to the elements of the 
    original dataset (subset_ds[idx]) which are tuples of (seq, target)."""
    return (np.array([np.array(subset_ds[idx][0]) for idx in range(len(subset_ds))], dtype=object),
            np.array([np.array(subset_ds[idx][1]) for idx in range(len(subset_ds))], dtype=object))


def yield_chars(path):
    with open(path) as f:
        for line in f:
            for char in line:
                yield char


def collate_fn(samples: list):
    X, Y = zip(*samples)
    X = torch.nn.utils.rnn.pad_sequence([torch.Tensor(x) for x in X], batch_first=True)
    Y = torch.nn.utils.rnn.pad_sequence([torch.Tensor(y) for y in Y], batch_first=True)
    return X, Y


def get_dataloaders(cfg, rng):
    print("Problem:", cfg.data.problem_name)

    print("Building vocabulary...")
    vocabulary = build_vocab_from_iterator(
        yield_chars(os.path.join(hydra.utils.get_original_cwd(),
                                 f"../data/external/mathematics_dataset-v1.0/train-{cfg.data.problem_difficulty}/{cfg.data.problem_name}.txt")))
    print(f"Built vocabulary with {len(vocabulary)} terms")
    ds = MathematicsDataset(cfg, transform=VocabTransform(vocabulary))

    perc_valid = 0.2
    size_train, size_valid = round(len(ds) * (1 - perc_valid)), round(len(ds) * perc_valid)
    train_ds, valid_ds = torch.utils.data.random_split(ds, [size_train, size_valid], generator=rng)

    train_X, _ = to_numpy_subset(train_ds)
    bucket_batch_sampler_train = BucketBatchSampler(train_X, cfg.train.batch_size)  # <-- does not store X
    train_dataloader = DataLoader(train_ds, batch_sampler=bucket_batch_sampler_train, shuffle=False,
                                  num_workers=1, drop_last=False, collate_fn=collate_fn)
    
    valid_X, _ = to_numpy_subset(valid_ds)
    bucket_batch_sampler_valid = BucketBatchSampler(valid_X, cfg.train.batch_size)
    valid_dataloader = DataLoader(valid_ds, batch_sampler=bucket_batch_sampler_valid, shuffle=False,
                                  num_workers=1, drop_last=False, collate_fn=collate_fn)
    
    return train_dataloader, valid_dataloader, vocabulary
