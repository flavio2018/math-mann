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
import torch.nn.modules as M

import logging


class DynamicNeuralTuringMachine(nn.Module):
    def __init__(self, memory, controller_hidden_state_size, controller_input_size, controller_output_size=10):
        super(DynamicNeuralTuringMachine, self).__init__()
        self.add_module("memory", memory)
        full_controller_input_size = controller_input_size + memory.overall_memory_size
        self.controller = M.GRUCell(input_size=full_controller_input_size, hidden_size=controller_hidden_state_size)
        self.W_output = nn.Parameter(torch.zeros(controller_output_size, controller_hidden_state_size))
        self.b_output = nn.Parameter(torch.zeros(controller_output_size, 1))

        self._init_parameters(init_function=nn.init.xavier_uniform_)

    def forward(self, x, num_addressing_steps=1):

        dummy_memory_reading = torch.zeros((self.memory.overall_memory_size, x.shape[1]))

        if num_addressing_steps < 1:
            raise ValueError(f"num_addressing_steps should be at least 1, received: {num_addressing_steps}")

        with torch.no_grad():
            logging.debug(f"{self.controller_hidden_state.isnan().any()=}")
            logging.debug(f"{self.controller_hidden_state.mean()=}")
            logging.debug(f"{self.controller_hidden_state.max()=}")
            logging.debug(f"{self.controller_hidden_state.min()=}")

        for __ in range(num_addressing_steps):
            # memory_reading = self.memory.read(self.controller_hidden_state)
            # self.memory.update(self.controller_hidden_state, x)
            self.controller_hidden_state = self.controller(torch.cat((x, dummy_memory_reading)).T,
                                                           self.controller_hidden_state.T).T.detach()
            # ^ TODO very hacky solution, should be improved

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

    def reshape_and_reset_hidden_states(self, batch_size, device):
        with torch.no_grad():
            controller_hidden_state_size = self.W_output.shape[1]
        self.register_buffer("controller_hidden_state", torch.zeros(size=(controller_hidden_state_size, batch_size)))
        self.controller_hidden_state = self.controller_hidden_state.to(device)
