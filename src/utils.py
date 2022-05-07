from datetime import datetime
import logging
import numpy as np
import torch
import random
import inspect


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


def configure_reproducibility(seed):
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
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    rng = torch.Generator()
    rng.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(mode=True, warn_only=True)

    return rng


def config_run(loglevel, run_name, seed):
    caller_name = inspect.stack()[1].function
    run_name = "_".join([caller_name, get_str_timestamp(), run_name])

    configure_logging(loglevel, run_name)

    # mlflow.set_tracking_uri("file:../logs/mlruns/")
    # mlflow.set_experiment(experiment_name="dntm_pmnist")

    writer = None  # SummaryWriter(log_dir=f"../logs/tensorboard/{run_name}")

    device = torch.device("cuda", 0)
    if loglevel == 'DEBUG':
        torch.autograd.set_detect_anomaly(True)

    rng = configure_reproducibility(seed)

    return device, rng, run_name, writer


def seed_worker(worker_id):
    """This is specifically for DataLoaders workers initialization.
    See https://pytorch.org/docs/stable/notes/randomness.html"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SimpleEarlyStopping:
    def __init__(self, patience, minimal_improvement, run_name):
        self.best_train_loss = 0
        self.steps_since_last_improvement = 0
        self.patience = patience
        self.minimal_improvement = minimal_improvement
        self.path = f"../models/{run_name}.pth"

    def early_stop(self, train_loss_value, model):
        if self.steps_since_last_improvement > self.patience:
            return True

        elif (self.best_train_loss == 0) or ((self.best_train_loss - train_loss_value) > self.minimal_improvement):
            self.steps_since_last_improvement = 0
            self.best_train_loss = train_loss_value

            logging.info(f"Saving the model")
            torch.save(model.state_dict(), self.path)

        else:
            self.steps_since_last_improvement += 1

        return False
