import pytest
import torch

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine, DynamicNeuralTuringMachineMemory


def _init_dntm_memory_parameters():
    n_locations = 100
    content_size = 100
    address_size = 20
    return {
        "n_locations": n_locations,
        "content_size": content_size,
        "address_size": address_size,
    }


def _mock_controller_input():
    controller_input_size = 100
    return torch.ones((controller_input_size, 1))


def _mock_controller_hidden_state():
    memory_parameters = _init_dntm_memory_parameters()
    return torch.randn((memory_parameters["n_locations"], 1))


def test_dntm_memory_content_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=100)
    dntm_memory.address_memory(mock_hidden_state)
    memory_contents = dntm_memory.read()
    assert memory_contents.shape == (
        memory_parameters["content_size"] + memory_parameters["address_size"], 1)


def test_dntm_memory_address_vector_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=100)
    dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.shape == (
        memory_parameters["n_locations"], 1)


def test_dntm_memory_address_vector_sum_to_one():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=100)
    dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.sum().item() == pytest.approx(1)
