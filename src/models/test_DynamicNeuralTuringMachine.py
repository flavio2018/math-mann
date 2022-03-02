import pytest
import torch

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine, DynamicNeuralTuringMachineMemory


CONTROLLER_INPUT_SIZE = 150


def _init_dntm_memory_parameters():
    n_locations = 100
    content_size = 120
    address_size = 20
    return {
        "n_locations": n_locations,
        "content_size": content_size,
        "address_size": address_size,
    }


def _mock_controller_input():
    controller_input_size = CONTROLLER_INPUT_SIZE
    return torch.ones((controller_input_size, 1))


def _mock_controller_hidden_state():
    memory_parameters = _init_dntm_memory_parameters()
    return torch.randn((memory_parameters["n_locations"], 1))


def test_dntm_memory_reading_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        memory_reading = dntm_memory.read()
    assert memory_reading.shape == (
        memory_parameters["content_size"] + memory_parameters["address_size"], 1)


def test_dntm_memory_address_vector_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.shape == (
        memory_parameters["n_locations"], 1)


def test_dntm_memory_address_vector_sum_to_one():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.sum().item() == pytest.approx(1)


def test_dntm_memory_contents_shape_doesnt_change_after_update():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        memory_contents_before_update = dntm_memory.memory_contents
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
    assert dntm_memory.memory_contents.shape == memory_contents_before_update.shape


def test_dntm_memory_is_zeros_after_reset():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
        dntm_memory.reset_memory_content()
    assert (dntm_memory.memory_contents == 0).all()
