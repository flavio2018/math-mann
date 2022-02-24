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


# TODO check all parameters shapes
# TODO implement unused memory slot
class DynamicNeuralTuringMachine(nn.Module):
    def __init__(self, memory, controller_hidden_state_size, controller_input_size):
        super(DynamicNeuralTuringMachine, self).__init__()
        self.add_module("memory", memory)
        self.controller = M.GRUCell(input_size=controller_input_size, hidden_size=controller_hidden_state_size)
        self.register_buffer("controller_hidden_state", torch.empty(size=(controller_hidden_state_size, 1)))

    def forward(self, x, num_addressing_steps=1):
        for __ in range(num_addressing_steps):
            self.memory.address_memory(self.controller_hidden_state)
            content_vector = self.memory.read()
            print(f"{content_vector.shape=}")
            self.controller_hidden_state = self.controller(torch.cat((x, content_vector)),
                                                           self.controller_hidden_state)
            self.memory.update(self.controller_hidden_state, x)  
        return self.controller_hidden_state  # TODO define output


class DynamicNeuralTuringMachineMemory(nn.Module):
    def __init__(self, n_locations, content_size, address_size, controller_input_size):
        """Instantiate the memory.
        n_locations: number of memory locations
        content_size: size of the content part of memory locations
        address_size: size of the address part of memory locations"""
        super(DynamicNeuralTuringMachineMemory, self).__init__()
        
        self.register_buffer("memory_contents", torch.zeros(size=(n_locations, content_size)))
        self.memory_addresses = nn.Parameter(torch.zeros(size=(n_locations, address_size)), requires_grad=True)
        self.overall_memory_size = content_size + address_size

        # query vector MLP parameters (W_k, b_k)
        self.W_query = nn.Parameter(torch.zeros(size=(self.overall_memory_size, n_locations)), requires_grad=True)
        self.b_query = nn.Parameter(torch.zeros(size=(self.overall_memory_size, 1)), requires_grad=True)

        # sharpening parameters (u_beta, b_beta)
        self.u_sharpen = nn.Parameter(torch.zeros(size=(1, n_locations)), requires_grad=True)
        self.b_sharpen = nn.Parameter(torch.zeros(1), requires_grad=True)

        # LRU parameters (u_gamma, b_gamma)
        self.u_lru = nn.Parameter(torch.zeros_like(self.u_sharpen))
        self.b_lru = nn.Parameter(torch.zeros_like(self.b_sharpen))
        self.register_buffer("exp_mov_avg_similarity", torch.zeros(size=(n_locations, 1)))

        # erase parameters (generate e_t)
        self.W_erase = nn.Parameter(torch.zeros(size=(content_size, n_locations)), requires_grad=True)
        self.b_erase = nn.Parameter(torch.zeros(size=(n_locations, 1)), requires_grad=True)

        # writing parameters (W_m, W_h, alpha)
        self.W_content_hidden = nn.Parameter(torch.zeros(size=(content_size, n_locations)), requires_grad=True)
        self.W_content_input = nn.Parameter(torch.zeros(size=(content_size, n_locations)), requires_grad=True)
        self.u_content_alpha = nn.Parameter(torch.zeros(size=(n_locations+controller_input_size, 1)), requires_grad=True)
        self.b_content_alpha = nn.Parameter(torch.zeros(size=(n_locations+controller_input_size, 1)), requires_grad=True)
        
        self.address_vector = None
        self._init_parameters()

    def read(self):
        if self.address_vector is None:
            raise RuntimeError("The memory cannot be read before being addressed. Every read operation should be "
                               "preceded by an address operation.")
        return self._full_memory_view().T @ self.address_vector

    def update(self, controller_hidden_state, controller_input):
        erase_vector = self.W_erase @ controller_hidden_state + self.b_erase
        alpha = self.u_content_alpha @ torch.cat((controller_hidden_state, controller_input), dim=1) + self.b_content_alpha
        candidate_content_vector = F.relu(self.W_content_hidden @ controller_hidden_state + alpha * self.W_content_input @ controller_input)

        for j in range(self.memory_contents.shape[0]):
            self.memory_contents[j, :] = ((1 - self.address_vector[j] * erase_vector) * self.memory_contents[j,:]
                                          + self.address_vector[j] * candidate_content_vector)

    def address_memory(self, controller_hidden_state):
        query = self.W_query @ controller_hidden_state + self.b_query
        sharpening_beta = F.softplus(self.u_sharpen @ controller_hidden_state + self.b_sharpen) + 1
        similarity_vector = self._compute_similarity(query, sharpening_beta)
        print(f"{similarity_vector.shape=}")
        lru_similarity_vector = self._apply_lru_addressing(similarity_vector, controller_hidden_state)
        self.address_vector = lru_similarity_vector

    def _full_memory_view(self):
        return torch.cat((self.memory_addresses, self.memory_contents), dim=1)

    def _compute_similarity(self, query, sharpening_beta):
        """Compute the sharpened cosine similarity vector between the query and the memory locations."""
        full_memory_view = self._full_memory_view()
        similarity_vector = []
        for j in range(self.memory_contents.shape[0]):
            similarity_value = sharpening_beta * F.cosine_similarity(full_memory_view[j, :], query.T, eps=1e-7)
            similarity_vector.append([similarity_value])  # we want a column vector so we create a list of lists
        return torch.tensor(similarity_vector)

    def _apply_lru_addressing(self, similarity_vector, controller_hidden_state):
        """Apply the Least Recently Used addressing mechanism. This shifts the addressing towards positions 
        that have not been recently read or written."""
        lru_gamma = F.sigmoid(self.u_lru @ controller_hidden_state + self.b_lru)
        print(f"{lru_gamma.shape=}")
        lru_similarity_vector = F.softmax(similarity_vector - lru_gamma * self.exp_mov_avg_similarity)
        print(f"{lru_similarity_vector.shape=}")
        self.exp_mov_avg_similarity = 0.1 * self.exp_mov_avg_similarity + 0.9 * similarity_vector
        print(f"{self.exp_mov_avg_similarity.shape=}")
        return lru_similarity_vector
    
    def _init_parameters(self):
        for name, parameter in self.named_parameters():
            if len(parameter.shape) > 1:
                print("Initializing parameter", name)
                nn.init.xavier_uniform_(parameter, gain=1)
        # TODO check params initialization method in paper
        # TODO check gain value based on activation function (i.e. using nn.init.calculate_gain)

    def reset_memory_content(self):
        """This method exists to implement the memory reset at the beginning of each episode.
        This logic should be implemented outside the model (e.g. using the helper function defined below)."""
        self.memory_content.fill_(0)
        # self.memory_content = torch.zeros(size=self.memory_content.shape)  # alternative

    def forward(self, x):
        raise RuntimeError("It makes no sense to call the memory module on its own. "
                           "The module should be accessed by the controller "
                           "either by addressing, reading or updating the memory.")
