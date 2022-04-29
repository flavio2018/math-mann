"""This script trains a DNTM on the PMNIST task."""
import torch.nn
import logging
import os
import hydra
import omegaconf
import wandb

from src.utils import seed_worker, configure_reproducibility
from src.data.perm_seq_mnist import get_dataset
from src.models.train_dntm_utils import build_model

import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchmetrics.classification import Accuracy
from torchvision.utils import make_grid
from src.models.pytorchtools import EarlyStopping


@hydra.main(config_path="../../conf", config_name="mnist")
def click_wrapper(cfg):
    train_and_test_dntm_smnist(cfg)


def train_and_test_dntm_smnist(cfg):
    device = torch.device("cuda", 0)
    rng = configure_reproducibility(cfg.run.seed)

    wandb.config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="dntm_mnist", entity="flapetr")
    wandb.run.name = cfg.run.codename

    train, test = get_dataset(cfg.data.permute, cfg.run.seed)
    train.data, train.targets = train.data[:cfg.data.num_train], train.targets[:cfg.data.num_train]
    test.data, test.targets = test.data[:cfg.data.num_test], test.targets[:cfg.data.num_test]

    # obtain training indices that will be used for validation
    valid_size = 0.2
    num_train = len(train)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_data_loader = DataLoader(train,
                                   batch_size=cfg.train.batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=train_sampler,
                                   num_workers=0,
                                   generator=rng)  # reproducibility

    valid_data_loader = DataLoader(train,
                                   batch_size=cfg.train.batch_size,
                                   shuffle=False,
                                   worker_init_fn=seed_worker,
                                   sampler=valid_sampler,
                                   num_workers=0,
                                   generator=rng)  # reproducibility

    model = build_model(cfg.model, device)

    loss_fn = torch.nn.NLLLoss()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    early_stopping = EarlyStopping(verbose=True,
                                   path=os.path.join(hydra.utils.get_original_cwd(),
                                                     f"../models/checkpoints/{cfg.run.codename}.pth"),
                                   trace_func=logging.info,
                                   patience=cfg.train.patience)


    # training
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(cfg.train.epochs):
        logging.info(f"Epoch {epoch}")

        train_loss, train_accuracy = training_step(device, model, loss_fn, opt,
                                                   train_data_loader, epoch, cfg.train.batch_size)
        valid_loss, valid_accuracy = valid_step(device, model, loss_fn, valid_data_loader)

        wandb.log({'loss_training_set': train_loss,
                   'loss_validation_set': valid_loss})
        wandb.log({'acc_training_set': train_accuracy,
                   'acc_validation_set': valid_accuracy})
        log_weights_gradient(model, wandb)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    # testing
    del train_data_loader
    test_data_loader = DataLoader(test, batch_size=cfg.train.batch_size)

    logging.info("Starting testing phase")
    test_step(device, model, test_data_loader)


def valid_step(device, model, loss_fn, valid_data_loader):
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        model.memory.reset_memory_content()
        model.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        model.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

        batch_accuracy = valid_accuracy(output.argmax(axis=0), targets)
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_data_loader.sampler)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def training_step(device, model, loss_fn, opt, train_data_loader, epoch, batch_size):
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):

        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            mnist_batch_img = wandb.Image(make_grid(mnist_images.reshape(batch_size, 1, 28, -1)))
            wandb.log({f"Training data batch {batch_i}, epoch {epoch}": mnist_batch_img})

        logging.debug(f"Resetting the memory")
        model.memory.reset_memory_content()
        model.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        model.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)
        model.controller_hidden_state = model.controller_hidden_state.detach()

        # if (epoch == 0) and (batch_i == 0):
        #     mocked_input = torch.ones(size=(1, mnist_images.shape[0]), device="cuda")
        #     hidden_state, output = model(mocked_input)
        #     writer.add_graph(model, mocked_input)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))

        if batch_i == 0:
            columns = ["Predictions", "Targets"]
            data = zip([str(p.item()) for p in output.argmax(axis=0)],
                       [str(t.item()) for t in targets])
            data = [list(row) for row in data]
            table = wandb.Table(data=data, columns=columns)
            wandb.log({"First batch preds vs targets": table})

        logging.debug(f"Computing loss value")
        logging.debug(f"{targets=}")
        loss_value = loss_fn(output.T, targets)
        # writer.add_scalar("per-batch_loss/train", loss_value, batch_i)
        epoch_loss += loss_value.item() * mnist_images.size(0)

        logging.debug(f"Computing gradients")
        loss_value.backward()

        logging.debug(f"Running optimization step")
        opt.step()

        batch_accuracy = train_accuracy(output.argmax(axis=0), targets)

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


def test_step(device, model, test_data_loader):
    test_accuracy = Accuracy().to(device)

    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        model.memory.reset_memory_content()
        model.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        model.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)
        model.controller_hidden_state = model.controller_hidden_state.detach()

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = model(pixels.view(1, -1))

        batch_accuracy = test_accuracy(output.argmax(axis=0), targets)
    test_accuracy = test_accuracy.compute()
    wandb.log({"test_accuracy": test_accuracy.item()})


if __name__ == "__main__":
    click_wrapper()
