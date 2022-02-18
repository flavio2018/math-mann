import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules as M


class DynamicNeuralTuringMachine(nn.Module):
    def __init__(self, memory):
        self.memory = memory
        self.controller = M.GRUCell()  # TODO fill params
        self.controller_hidden_state = torch.zeros()  # TODO fill params


    def forward(self, x):
        self.memory.address_memory(self.controller_hidden_state)
        content_vector = self.memory.read(self.controller_hidden_state)  
        self.controller_hidden_state = self.controller(nn.concat([x, content_vector]), self.controller_hidden_state)  # TODO check op
        self.memory.update(self.controller_hidden_state, x)  
        return self.controller_hidden_state  # TODO define output


class DynamicNeuralTuringMachineMemory(nn.Module):
    def __init__(self, n_locations, content_size, address_size):
        """Instantiate the memory.
        n_locations: number of memory locations
        content_size: size of memory locations"""
        
        self.memory_contents = torch.zeros(size=(n_locations, content_size))
        self.memory_addresses = torch.zeros(size=(n_locations, address_size), requires_grad=True)
        
        # TODO refactor layers into low-level matrices/vectors
        self.query_mlp= M.linear(in_features=(content_size + address_size), out_features=n_locations, bias=True)
        self.sharpening_mlp = M.linear(in_features=n_locations, out_features=1, bias=True)
        self.lru_mlp = M.linear(in_features=n_locations, out_features=1, bias=True)
        self.erase_mlp = M.linear(in_features=n_locations, out_features=content_size, bias=True)
        self.candidate_hidden_mlp = M.linear(in_features=n_locations, out_features=content_size, bias=False)
        self.candidate_input_mlp = M.linear(in_features=n_locations, out_features=content_size, bias=False)
        self.candidate_alpha_mlp = M.linear(in_features=(n_locations + input_size), out_features=1, bias=True)  # TODO input_size?
        
        self.address_vector = None
    
    def read(self, controller_hidden_state):
        return nn.concat(self.memory_addresses, self.memory_contents) @ self.address_vector  # TODO check operation

    def update(self, controller_hidden_state, controller_input):
        erase_vector = self.erase_mlp(controller_hidden_state)
        alpha = self.candidate_alpha_mlp(nn.concat(controller_hidden_state, controller_input))
        candidate_content_vector = F.relu(self.candidate_input_mlp(controller_hidden_state) + alpha * self.candidate_input_mlp(controller_input))
        for j in range(self.memory_contents.shape[0]):
            self.memory_contents[j, :] = (1 - self.address_vector[j] * erase_vector) * self.memory_contents[j,:] + self.address_vector[j] * candidate_content_vector

    def address_memory(self, controller_hidden_state):
        query = self.query_mlp(controller_hidden_state)
        sharpening_beta = F.softplus(self.sharpening_mlp(controller_hidden_state)) + 1
        similarity_vector = self._compute_similarity(query, sharpening_beta)
        lru_similarity_vector = self._apply_lru_addressing(similarity_vector)
        self.address_vector = lru_similarity_vector

    def _compute_similarity(self, query, sharpening_beta):
        """Compute the sharpened cosine similarity vector between the query and the memory locations."""
        full_memory_view = nn.concat(self.memory_addresses, self.memory_contents)  # TODO check op
        similarity_vector = []
        for j in range(self.memory_contents.shape[0]):
            similarity_vector.append(sharpening_beta * F.cosine_similarity(full_memory_view[j, :], query, eps=1e-7))
        return nn.tensor(similarity_vector)  # TODO check cast

    def _apply_lru_addressing(self, similarity_vector)
        """Apply the Least Recently Used addressing mechanism. This shifts the addressing towards positions 
        that have not been recently read or written."""
        lru_gamma = F.sigmoid(self.lru_mlp(controller_hidden_state))
        lru_similarity_vector = F.softmax(similarity_vector - lru_gamma * self.exp_mov_avg_similarity)
        self.exp_mov_avg_similarity = 0.1 * self.exp_mov_avg_similarity + 0.9 * similarity_vector
        return lru_similarity_vector
    
    def forward(self, x)
        pass  # TODO

    # TODO define some hook that erases the memory content at the beginning of each episode

