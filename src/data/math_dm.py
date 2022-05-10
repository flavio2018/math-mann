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

    def numpy(self):
        return (np.array([np.array(seq) for seq in self[:][0]], dtype=object),
                np.array([np.array(target_seq) for target_seq in self[:][1]], dtype=object))


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


def get_dataloader(cfg):
    print("Problem:", cfg.data.problem_name)

    print("Building vocabulary...")
    vocabulary = build_vocab_from_iterator(
        yield_chars(os.path.join(hydra.utils.get_original_cwd(),
                                 f"../data/external/mathematics_dataset-v1.0/train-{cfg.data.problem_difficulty}/{cfg.data.problem_name}.txt")))
    print(f"Built vocabulary with {len(vocabulary)} terms")
    ds = MathematicsDataset(cfg, transform=VocabTransform(vocabulary))

    X, y = ds.numpy()
    bucket_batch_sampler = BucketBatchSampler(X, 16)  # <-- does not store X

    return DataLoader(ds, batch_sampler=bucket_batch_sampler, shuffle=False,
                      num_workers=2, drop_last=False, collate_fn=collate_fn), vocabulary
