import hydra
import torch

from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from src.models.train_dntm_utils import build_dntm, get_digit_string_repr
from src.data.perm_seq_mnist import get_dataset


@hydra.main("../../conf", "viz_addr")
def main(cfg):
    device = torch.device("cuda", 0)
    dntm = build_dntm(cfg.model, device)
    
    _, test = get_dataset(permute=False, seed=0)
    test.data, test.targets = test.data[:10], test.targets[:10]
    test_data_loader = DataLoader(test, batch_size=1)
    accuracy = Accuracy().to(device)

    dntm.eval()
    num_batch = 0
    for batch, targets in test_data_loader:
        print("Batch", num_batch, "Target", targets)
        print(get_digit_string_repr(batch[0, :, :])) 
        batch_size, sequence_len, feature_size = batch.shape
        dntm.prepare_for_batch(batch, device)
        batch, targets = batch.to(device), targets.to(device)
        _, output = dntm(batch)
        print(output)
        accuracy.update(output.T, targets)
        num_batch += 1
    print("Test accuracy:", accuracy.compute())


if __name__ == '__main__':
    main()
