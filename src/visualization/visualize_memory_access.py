import hydra
import torch

from src.models.train_dntm_utils import build_dntm
from src.data.perm_seq_mnist import get_dataset, get_dataloaders
from src.utils import configure_reproducibility


@hydra.main("../../conf", "viz_addr")
def main(cfg):
    device = torch.device("cuda", 0)
    model = build_dntm(cfg.model, device)
    rng = configure_reproducibility(cfg.run.seed)

    train_dataloader, valid_dataloader = get_dataloaders(cfg, rng)

    for batch, targets in train_dataloader: 
        batch_size, sequence_len, feature_size = batch.shape
        model.prepare_for_batch(batch, device)
        batch, targets = batch.to(device), targets.to(device)

if __name__ == '__main__':
    main()
