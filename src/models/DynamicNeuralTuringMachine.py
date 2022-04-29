"""This script contains the implementation of a Dynamic-Neural Turing Machine.

By convention, tensors whose name starts with a 'W' are bidimensional (i.e. matrices), 
while tensors whose name starts with a 'u' or a 'b' are one-dimensional (i.e. vectors).
Usually, these parameters are part of linear transformations implementing a multi-input perceptron,
thereby representing the weights and biases of these operations.

The choice that was made in this implementation is to decouple the external memory of the model
as a separate PyTorch Module. The full D-NTM model is thus composed of a controller module and a 
memory module."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging


class DynamicNeuralTuringMachine(nn.Module):
    def __init__(self, memory, controller_hidden_state_size, controller_input_size, controller_output_size=10):
        super(DynamicNeuralTuringMachine, self).__init__()
        self.add_module("memory", memory)
        self.controller = CustomGRU(input_size=controller_input_size,
                                    hidden_size=controller_hidden_state_size,
                                    memory_size=memory.overall_memory_size)
        self.W_output = nn.Parameter(torch.zeros(controller_output_size, controller_hidden_state_size))
        self.b_output = nn.Parameter(torch.zeros(controller_output_size, 1))

        self._init_parameters(init_function=nn.init.xavier_uniform_)

    def forward(self, x, num_addressing_steps=1):

        if num_addressing_steps < 1:
            raise ValueError(f"num_addressing_steps should be at least 1, received: {num_addressing_steps}")

        with torch.no_grad():
            logging.debug(f"{self.controller_hidden_state.isnan().any()=}")
            logging.debug(f"{self.controller_hidden_state.mean()=}")
            logging.debug(f"{self.controller_hidden_state.max()=}")
            logging.debug(f"{self.controller_hidden_state.min()=}")

        for __ in range(num_addressing_steps):
            memory_reading = self.memory.read(self.controller_hidden_state)
            self.memory.update(self.controller_hidden_state, x)
            self.controller_hidden_state = self.controller(x, self.controller_hidden_state,
                                                           memory_reading)

            output = F.log_softmax(self.W_output @ self.controller_hidden_state + self.b_output, dim=0)
        return self.controller_hidden_state, output

    def _init_parameters(self, init_function):
        logging.info(f"Initialization method: {init_function.__name__}")
        # Note: the initialization method is not specified in the original paper
        for name, parameter in self.named_parameters():
            if len(parameter.shape) > 1:
                logging.info(f"Initializing parameter {name}")
                if name in ("memory_addresses", "W_query", "b_query"):
                    init_function(parameter, gain=1)
                elif name in ("u_sharpen", "W_content_hidden", "W_content_input"):
                    init_function(parameter, gain=torch.nn.init.calculate_gain("relu"))
                elif name == "u_lru":
                    init_function(parameter, gain=torch.nn.init.calculate_gain("sigmoid"))
                else:
                    init_function(parameter)
                logging.debug(f"{name}: {parameter}")
            if name == 'b_output':
                logging.info("Initializing bias b_output")
                torch.nn.init.constant_(parameter, 0.1)
                logging.debug(f"{name}: {parameter}")

    def prepare_for_batch(self, batch, device):
        self.memory._reset_memory_content()
        self._reshape_and_reset_hidden_states(batch_size=batch.shape[0], device=device)
        self.memory._reshape_and_reset_exp_mov_avg_sim(batch_size=batch.shape[0], device=device)
        self.controller_hidden_state = self.controller_hidden_state.detach()

    def _reshape_and_reset_hidden_states(self, batch_size, device):
        with torch.no_grad():
            controller_hidden_state_size = self.W_output.shape[1]
        self.register_buffer("controller_hidden_state", torch.zeros(size=(controller_hidden_state_size, batch_size)))
        self.controller_hidden_state = self.controller_hidden_state.to(device)


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, device=None):
        super().__init__()
        # input-hidden parameters
        self.W_ir = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.W_iz = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.W_in = torch.nn.Parameter(torch.zeros((hidden_size, input_size)))
        self.b_ir = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_iz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_in = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

        # hidden-hidden parameters
        self.W_hr = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.W_hz = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.W_hn = torch.nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.b_hr = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_hz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_hn = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

        # memory-hidden parameters
        self.W_mr = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.W_mz = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.W_mn = torch.nn.Parameter(torch.zeros((hidden_size, memory_size)))
        self.b_mr = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_mz = torch.nn.Parameter(torch.zeros((hidden_size, 1)))
        self.b_mn = torch.nn.Parameter(torch.zeros((hidden_size, 1)))

    def forward(self, input, hidden, memory_reading):
        sigm = torch.nn.Sigmoid()
        tanh = torch.nn.Tanh()

        r = sigm(self.W_ir @ input + self.b_ir +
                 self.W_hr @ hidden + self.b_hr +
                 self.W_mr @ memory_reading + self.b_mr)
        z = sigm(self.W_iz @ input + self.b_iz +
                 self.W_hz @ hidden + self.b_hz +
                 self.W_mz @ memory_reading + self.b_mz)
        n = tanh(self.W_in @ input + self.b_in +
                 self.W_mn @ memory_reading + self.b_mn
                 + r * (self.W_hn @ hidden + self.b_hn))
        return (1 - z) * n + z * hidden
