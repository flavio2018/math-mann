"""This script trains a DNTM on the PMNIST task."""
import torch.nn
import logging
import os
import hydra
import omegaconf
import wandb

from src.data.perm_seq_mnist import get_dataloaders
from src.models.train_dntm_utils import build_model
from src.utils import configure_reproducibility
from src.wandb_utils import log_weights_gradient, log_preds_and_targets

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

    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)
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

        train_loss, train_accuracy = training_step(device, model, loss_fn, opt, train_dataloader, epoch)
        valid_loss, valid_accuracy = valid_step(device, model, loss_fn, valid_dataloader)

        wandb.log({'loss_training_set': train_loss,
                   'loss_validation_set': valid_loss})
        wandb.log({'acc_training_set': train_accuracy,
                   'acc_validation_set': valid_accuracy})
        log_weights_gradient(model)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break


def valid_step(device, model, loss_fn, valid_data_loader):
    valid_accuracy = Accuracy().to(device)
    valid_epoch_loss = 0
    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(valid_data_loader):
        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, output = model(mnist_images.T.unsqueeze(dim=-1))

        loss_value = loss_fn(output.T, targets)
        valid_epoch_loss += loss_value.item() * mnist_images.size(0)

        batch_accuracy = valid_accuracy(output.argmax(axis=0), targets)
    valid_accuracy_at_epoch = valid_accuracy.compute()
    valid_epoch_loss /= len(valid_data_loader.sampler)
    valid_accuracy.reset()
    return valid_epoch_loss, valid_accuracy_at_epoch


def training_step(device, model, loss_fn, opt, train_data_loader, epoch):
    train_accuracy = Accuracy().to(device)

    epoch_loss = 0
    model.train()
    for batch_i, (mnist_images, targets) in enumerate(train_data_loader):
        batch_size = len(mnist_images)
        logging.info(f"MNIST batch {batch_i}")
        model.zero_grad()

        if (epoch == 0) and (batch_i == 0):
            mnist_batch_img = wandb.Image(make_grid(mnist_images.reshape(batch_size, 1, 28, -1)))
            wandb.log({f"Training data batch {batch_i}, epoch {epoch}": mnist_batch_img})

        model.prepare_for_batch(mnist_images, device)

        # if (epoch == 0) and (batch_i == 0):
        #     mocked_input = torch.ones(size=(1, mnist_images.shape[0]), device="cuda")
        #     hidden_state, output = model(mocked_input)
        #     writer.add_graph(model, mocked_input)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        _, output = model(mnist_images.T.unsqueeze(dim=-1))
        log_preds_and_targets(batch_i, output, targets)

        loss_value = loss_fn(output.T, targets)
        epoch_loss += loss_value.item() * mnist_images.size(0)

        loss_value.backward()
        opt.step()

        batch_accuracy = train_accuracy(output.argmax(axis=0), targets)

    accuracy_over_batches = train_accuracy.compute()
    epoch_loss /= len(train_data_loader.sampler)
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches


if __name__ == "__main__":
    click_wrapper()
