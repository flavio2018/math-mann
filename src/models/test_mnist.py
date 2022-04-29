import hydra
import logging
import wandb

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from src.data.perm_seq_mnist import get_dataset
from src.models.train_dntm_utils import build_model


@hydra.main(config_path="../../conf", config_name="test_mnist")
def test_mnist(cfg):
    device = torch.device("cuda", 0)
    _, test = get_dataset(cfg.data.permute, cfg.run.seed)
    test.data, test.targets = test.data[:cfg.data.num_test], test.targets[:cfg.data.num_test]
    test_data_loader = DataLoader(test, batch_size=cfg.train.batch_size)
    model = build_model(cfg.model, device)

    logging.info("Starting testing phase")
    test_step(device, model, test_data_loader)


def test_step(device, model, test_data_loader):
    test_accuracy = Accuracy().to(device)

    model.eval()
    for batch_i, (mnist_images, targets) in enumerate(test_data_loader):
        logging.info(f"MNIST batch {batch_i}")

        model.prepare_for_batch(mnist_images, device)

        mnist_images, targets = mnist_images.to(device), targets.to(device)

        for pixel_i, pixels in enumerate(mnist_images.T):
            __, output = model(pixels.view(1, -1))

        batch_accuracy = test_accuracy(output.argmax(axis=0), targets)
    test_accuracy = test_accuracy.compute()
    wandb.log({"test_accuracy": test_accuracy.item()})


if __name__ == '__main__':
    test_mnist()