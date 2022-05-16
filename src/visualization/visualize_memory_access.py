import hydra
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import torch
import seaborn as sns

import os

from src.models.train_dntm_utils import build_dntm
from src.data.perm_seq_mnist import get_dataset, get_dataloaders
from src.utils import configure_reproducibility


@hydra.main("../../conf", "viz_addr_local")
def main(cfg):
    device = torch.device("cuda", 0)

    print("Building model and loading data")
    model = build_dntm(cfg.model, device)
    rng = configure_reproducibility(cfg.run.seed)
    _, valid_dataloader = get_dataloaders(cfg, rng)

    window_size = 20  
    read_weights_matrix = np.zeros((cfg.model.n_locations, window_size))
    read_matrix_sequence = [read_weights_matrix]

    model.eval()
    for batch, targets in valid_dataloader:
        batch_size, sequence_len, feature_size = batch.shape
        model.prepare_for_batch(batch, device)
        batch, targets = batch.to(device), targets.to(device)

        print("Scanning one sequence")
        for i_seq in range(sequence_len):
            batch_element = batch[:, i_seq, :].reshape(feature_size, batch_size)
            controller_hidden_state, output = model.step_on_batch_element(batch_element)
            read_weights_matrix = slide_access_weights(read_weights_matrix)
            read_weights_matrix[:, -1] = model.memory.read_weights.squeeze().detach().cpu().numpy()
            read_matrix_sequence.append(read_weights_matrix)            
        break

    max_weigth, min_weigth = find_min_max_value(read_matrix_sequence, window_size)
    print(f"Maximum weight {max_weigth}, minimum weight {min_weigth}")

    print("Creating the GIF image")
    fig = plt.figure()
    sns.heatmap(read_matrix_sequence[0], vmin=min_weigth, vmax=max_weigth,
        xticklabels=False, yticklabels=False)


    def init():
        sns.heatmap(read_matrix_sequence[0], vmin=min_weigth, vmax=max_weigth,
            cbar=False, xticklabels=False, yticklabels=False)

    def animate(i):
        start_from=0
        sns.heatmap(read_matrix_sequence[start_from+i], vmin=min_weigth, vmax=max_weigth,
            cbar=False, xticklabels=False, yticklabels=False)
        plt.gca().set_title(str(start_from+i))

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=50, repeat = False)

    print("Saving file")
    savefile = os.path.join(hydra.utils.get_original_cwd(), "memory_access.gif")
    pillowwriter = animation.PillowWriter(fps=15)
    anim.save(savefile, writer=pillowwriter)


def slide_access_weights(weights_matrix):
    new_weights_matrix = np.zeros(weights_matrix.shape)
    new_weights_matrix[:, :-1] = weights_matrix[:, 1:]
    return new_weights_matrix


def find_min_max_value(sequence_read_weights, window_size):
    max_, min_ = -10, 10
    where_max, where_min = None, None

    for matrix_i, matrix in enumerate(sequence_read_weights):
        if matrix_i < window_size:
            continue
        if matrix.max() > max_:
            max_ = matrix.max()
            where_max = matrix_i
        if matrix.min() < min_:
            min_ = matrix.min()
            where_min = matrix_i

    print(f"Min element at position {where_min}, max element at position {where_max}")
    return max_, min_


if __name__ == '__main__':
    main()
