import torch
from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory
import logging


def read_write_consistency_regularizer(sequence_read_weights, sequence_write_weights, lambda_):
    """This method implements the first regularization term described in the paper.
    The regularization term uses all the read and write weights computed by the model
    processing a sequence, which should therefore be collected during training.

    The method assumes that read and write weights are column vectors, i.e. they have shape (1, N)
    and the batches of weights have shapes (k, N), where N is the number of memory locations and k
    is the sequence length."""
    term = torch.zeros(sequence_read_weights.shape[0])
    for t in range(sequence_read_weights.shape[0]):
        normalized_sum_of_write_weights_up_to_t = sequence_write_weights[:t+1, :].sum(axis=0).view(1, -1) / (t+1)
        scaled_product_with_read_weights = 1 - normalized_sum_of_write_weights_up_to_t.T @ sequence_read_weights[t, :].view(1, -1)
        norm = torch.linalg.matrix_norm(scaled_product_with_read_weights)
        term[t] = norm**2
    return lambda_ * term.sum()


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


