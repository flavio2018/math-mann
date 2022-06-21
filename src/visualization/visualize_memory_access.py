import hydra
import omegaconf
import matplotlib.pyplot as plt
from matplotlib import animation as mpl_animation
import numpy as np
import torch
import seaborn as sns
import wandb

import os

from src.models.train_dntm_utils import build_dntm
from src.data.perm_seq_mnist import get_dataset, get_dataloaders
from src.utils import configure_reproducibility


@hydra.main("../../conf", "viz_addr_local")
def main(cfg):
    device = torch.device("cuda", 0)
    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    wandb.init(project="dntm_mnist", entity="flapetr", mode=cfg.run.wandb_mode)
    wandb.run.name = cfg.run.codename
    for subconfig_name, subconfig_values in cfg_dict.items():
        if isinstance(subconfig_values, dict):
            wandb.config.update(subconfig_values)
        else:
            logging.warning(f"{subconfig_name} is not being logged.")

    text_table = wandb.Table(columns=["prediction", "target"])

    print("Building model and loading data")
    model = build_dntm(cfg.model, device)
    rng = configure_reproducibility(cfg.run.seed)
    _, valid_dataloader = get_dataloaders(cfg, rng)

    window_size = 30  
    read_weights_matrix = np.zeros((cfg.model.n_locations, window_size))
    write_weights_matrix = np.zeros((cfg.model.n_locations, window_size))
    read_weights_full = np.zeros((cfg.model.n_locations, 784))
    write_weights_full = np.zeros((cfg.model.n_locations, 784))
    read_matrix_sequence = [read_weights_matrix]
    write_matrix_sequence = [write_weights_matrix]
    
    model.eval()
    num_batch = 0
    for batch, targets in valid_dataloader:
        if num_batch < cfg.data.skip_n_batches:
            num_batch += 1
        else:
            batch_size, sequence_len, feature_size = batch.shape
            model.prepare_for_batch(batch, device)
            batch, targets = batch.to(device), targets.to(device)

            print("Scanning one sequence")
            for i_seq in range(sequence_len):
                batch_element = batch[:, i_seq, :].reshape(feature_size, batch_size)
                controller_hidden_state, output = model.step_on_batch_element(batch_element)
                read_weights_matrix = slide_access_weights(read_weights_matrix)
                write_weights_matrix = slide_access_weights(write_weights_matrix)
                read_weights_matrix[:, -1] = model.memory.read_weights.squeeze().detach().cpu().numpy()
                write_weights_matrix[:, -1] = model.memory.write_weights.squeeze().detach().cpu().numpy()
                read_matrix_sequence.append(read_weights_matrix)
                write_matrix_sequence.append(write_weights_matrix)

                read_weights_full = slide_access_weights(read_weights_full)
                write_weights_full = slide_access_weights(write_weights_full)
                read_weights_full[:, -1] = model.memory.read_weights.squeeze().detach().cpu().numpy()
                write_weights_full[:, -1] = model.memory.write_weights.squeeze().detach().cpu().numpy()

            target = targets.item()
            prediction = output.argmax(dim=0).item()
            text_table.add_data(prediction, target)
            break

    wandb.log({'predictions': text_table})
    save_heatmap(read_weights_full, read=True)
    save_heatmap(write_weights_full, read=False)
    save_animation(create_animation(read_matrix_sequence, window_size), read=True)
    save_animation(create_animation(write_matrix_sequence, window_size), read=False)    


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


def create_animation(matrix_sequence, window_size):
    print("Creating the GIF image")

    max_weigth, min_weigth = find_min_max_value(matrix_sequence, window_size)
    print(f"Maximum weight {max_weigth}, minimum weight {min_weigth}")

    fig = plt.figure()
    sns.heatmap(matrix_sequence[0], vmin=min_weigth, vmax=max_weigth,
        xticklabels=False, yticklabels=False)

    def init():
        sns.heatmap(matrix_sequence[0], vmin=min_weigth, vmax=max_weigth,
            cbar=False, xticklabels=False, yticklabels=False)

    def animate(i):
        start_from=0
        if i < window_size:
            labels=['']*(window_size) #  + list(range(range(start_from+i,start_from+i+window_size)))
        else:
            labels = range(start_from+i-window_size,start_from+i)
        sns.heatmap(matrix_sequence[start_from+i], vmin=min_weigth, vmax=max_weigth,
            cbar=False, xticklabels=labels, yticklabels=False)
        plt.gca().set_title(str(start_from+i))

    return mpl_animation.FuncAnimation(fig, animate, init_func=init, frames=50, repeat=False)


def save_animation(animation, read: bool):
    read_or_write = 'read' if read else 'write'
    
    print(f"Saving {read_or_write} animation")
    path = os.path.join(hydra.utils.get_original_cwd(), "../reports/figures/" f"memory_access_{read_or_write}.gif")
    pillowwriter = mpl_animation.PillowWriter(fps=10)
    animation.save(path, writer=pillowwriter)
    wandb.log({f"memory_access_{read_or_write}": wandb.Video(path, fps=4, format="gif")})


def save_heatmap(weights_full, read: bool):
    read_or_write = 'read' if read else 'write'
    
    print(f"Saving {read_or_write} heatmap")
    fig = plt.figure(figsize=(9, 3))
    sns.heatmap(weights_full, vmin=weights_full.min(), vmax=weights_full.max(),
                cbar=True, xticklabels=False, yticklabels=False)
    path = os.path.join(hydra.utils.get_original_cwd(), "../reports/figures/" f"memory_access_{read_or_write}_full.png")
    plt.savefig(path)
    wandb.log({f"memory_access_{read_or_write}_full": wandb.Image(path)})
    

if __name__ == '__main__':
    main()
