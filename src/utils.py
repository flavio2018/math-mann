from datetime import datetime
import logging
import numpy as np
import torch
import random


def configure_logging(loglevel, run_name):
    # assuming loglevel is bound to the string value obtained from the
    # command line argument. Convert to upper case to allow the user to
    # specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)

    logging.basicConfig(
        filename=f"../logs/{run_name}.log",
        level=numeric_level,
        format='%(levelname)s:%(filename)s:%(message)s',
        filemode="w")


def get_str_timestamp():
    return datetime.now().strftime("%Y%B%d_%H-%M-%S")


def configure_reproducibility(device, seed):
    """This is to try and ensure reproducibility, although it is known that it cannot be fully ensured across different
    PyTorch versions, CUDA and cuDNN versions and systems.

    Additionally to the instruction executed in this function, other operations must be done to try to ensure
    reproducibility. Specifically, randomness in the DataLoader PyTorch object creation must be taken into account,
    and known random behavior happening when using reccurrent and multi-head
    operations implemented in the CUBLAS library should be taken into account, e.g. by setting the environmental
    variable CUBLAS_WORKSPACE_CONFIG to ":4096:8".

    For reference, see:
    - https://discuss.pytorch.org/t/random-seed-initialization/7854/18
    - https://pytorch.org/docs/stable/notes/randomness.html
    """
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are suing GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(mode=True, warn_only=True)


def seed_worker(worker_id):
    """This is specifically for DataLoaders workers initialization.
    See https://pytorch.org/docs/stable/notes/randomness.html"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
