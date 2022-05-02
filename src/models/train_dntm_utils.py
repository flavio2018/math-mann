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


def build_model(model_conf, device):
    if model_conf.name == 'dntm':
        return build_dntm(model_conf, device)
    elif model_conf.name == 'mlp':
        return build_mlp(model_conf, device)
    else:
        return build_rnn(model_conf, device)


def build_dntm(model_conf, device):
    dntm_memory = DynamicNeuralTuringMachineMemory(
        n_locations=model_conf.n_locations,
        content_size=model_conf.content_size,
        address_size=model_conf.address_size,
        controller_input_size=model_conf.controller_input_size,
        controller_hidden_state_size=model_conf.controller_hidden_state_size
    )

    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=model_conf.controller_hidden_state_size,
        controller_input_size=model_conf.controller_input_size,
        controller_output_size=model_conf.controller_output_size
    ).to(device)

    if model_conf.ckpt is not None:
        logging.info(f"Reloading from checkpoint: {model_conf.ckpt}")
        state_dict = torch.load(model_conf.ckpt)
        batch_size_ckpt = state_dict['controller_hidden_state'].shape[1]
        dntm.memory._reset_memory_content()
        dntm._reshape_and_reset_hidden_states(batch_size=batch_size_ckpt, device=device)
        dntm.memory._reshape_and_reset_exp_mov_avg_sim(batch_size=batch_size_ckpt, device=device)
        dntm.memory.reshape_and_reset_read_write_weights(shape=state_dict['memory.read_weights'].shape)
        dntm.load_state_dict(state_dict)
    return dntm


class CustomLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, proj_size, device):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  proj_size=proj_size).to(device)

    def forward(self, batch):
        full_output, h_n_c_n = self.lstm(batch)
        last_output = full_output[-1, :, :].view((10, -1))  # select only the output for the last timestep and reshape to 2D
        log_soft_output = torch.nn.functional.log_softmax(last_output, dim=0)
        return h_n_c_n, log_soft_output

    def prepare_for_batch(self, batch, device):
        return


def build_rnn(model_conf, device):
    return CustomLSTM(input_size=model_conf.input_size,
                      hidden_size=model_conf.hidden_size,
                      proj_size=model_conf.output_size,
                      device=device)


class CustomMLP(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: str, output_size: int, device):
        """hidden_sizes should be a list of comma separated integers."""
        hidden_sizes = [int(size) for size in hidden_sizes.split(",")]
        super().__init__()
        hidden_sizes = [input_size] + hidden_sizes
        self.linear_layers = []
        for i in range(len(hidden_sizes) - 1):
            self.linear_layers += [torch.nn.Linear(in_features=hidden_sizes[i],
                                                   out_features=hidden_sizes[i+1],
                                                   device=device)]
        self.output_layer = torch.nn.Linear(in_features=hidden_sizes[-1], out_features=output_size, device=device)

    def forward(self, x):
        x = x.view(-1, 784)
        relu = torch.nn.ReLU()

        for layer in self.linear_layers:
            x = layer(x)
            x = relu(x)
        x = self.output_layer(x).T  # output shape is (-1, output_size)
        return None, torch.nn.functional.log_softmax(x, dim=0)

    def prepare_for_batch(self, batch, device):
        return


def build_mlp(model_conf, device):
    return CustomMLP(input_size=model_conf.input_size,
                     hidden_sizes=model_conf.hidden_sizes,
                     output_size=model_conf.output_size,
                     device=device)
