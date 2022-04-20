"""Use this script to visualize weights used to access memory."""
import hydra
import logging

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory
from src.data.perm_seq_mnist import get_dataset
from torch.utils.data import DataLoader
import torch

from src.utils import config_run


@hydra.main(config_path="../../conf", config_name="mem_weights")
def click_wrapper(cfg):
    inspect_mem_weights(cfg)


def inspect_mem_weights(cfg):
    _, _, run_name, writer = config_run(cfg.run.loglevel, cfg.run.run_name, seed=cfg.run.seed)

    n_locations = 12000
    device = torch.device("cpu")
    dntm = build_model(cfg.model.ckpt, address_size=8, content_size=16, controller_input_size=1, controller_output_size=10,
                       controller_hidden_state_size=100, device=device, n_locations=n_locations)

    _, test = get_dataset(permute=False, seed=cfg.run.seed)
    test.data, test.labels = test.data[:10], test.labels[:10]

    test_data_loader = DataLoader(test, batch_size=1)

    dntm.eval()
    for mnist_images, targets in test_data_loader:
        dntm.memory.reset_memory_content()
        dntm.reshape_and_reset_hidden_states(batch_size=mnist_images.shape[0], device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=mnist_images.shape[0], device=device)

        logging.debug(f"Moving image to GPU")
        mnist_images, targets = mnist_images.to(device), targets.to(device)

        logging.debug(f"Looping through image pixels")
        for pixel_i, pixels in enumerate(mnist_images.T):
            logging.debug(f"Pixel {pixel_i}")
            __, output = dntm(pixels.view(1, -1))
            logging.info(dntm.memory.read_weights)


def build_model(ckpt, address_size, content_size, controller_input_size, controller_output_size,
                controller_hidden_state_size, device, n_locations):
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=n_locations,
        content_size=content_size,
        address_size=address_size,
        controller_input_size=controller_input_size,
        controller_hidden_state_size=controller_hidden_state_size
    )
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=controller_hidden_state_size,
        controller_input_size=controller_input_size,
        controller_output_size=controller_output_size
    ).to(device)
    if ckpt is not None:
        logging.info(f"Reloading from checkpoint: {ckpt}")
        state_dict = torch.load(ckpt)
        batch_size_ckpt = state_dict['controller_hidden_state'].shape[1]
        dntm.memory.reset_memory_content()
        dntm.reshape_and_reset_hidden_states(batch_size=batch_size_ckpt, device=device)
        dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=batch_size_ckpt, device=device)
        dntm.memory.reshape_and_reset_read_write_weights(shape=state_dict['memory.read_weights'].shape)
        dntm.load_state_dict(state_dict)
    return dntm


if __name__ == "__main__":
    click_wrapper()
