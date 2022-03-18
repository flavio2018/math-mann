import pytest
import torch

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


CONTROLLER_INPUT_SIZE = 150
CONTROLLER_OUTPUT_SIZE = 10
BATCH_SIZE = 4


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
    return torch.ones((CONTROLLER_INPUT_SIZE, BATCH_SIZE))


def _mock_controller_hidden_state():
    memory_parameters = _init_dntm_memory_parameters()
    return torch.randn((memory_parameters["n_locations"], BATCH_SIZE))


def test_dntm_memory_reading_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        memory_reading = dntm_memory.read()
    assert memory_reading.shape == (
        memory_parameters["content_size"] + memory_parameters["address_size"], BATCH_SIZE)


def test_dntm_memory_address_vector_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.shape == (
        memory_parameters["n_locations"], BATCH_SIZE)


def test_dntm_memory_address_vector_contains_no_nan_values():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
    assert not dntm_memory.address_vector.isnan().any()


def test_dntm_memory_address_vector_sum_to_one():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
    assert dntm_memory.address_vector.sum().item() == pytest.approx(BATCH_SIZE)


def test_dntm_memory_contents_shape_doesnt_change_after_update():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        memory_contents_before_update = dntm_memory.memory_contents
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
    assert dntm_memory.memory_contents.shape == memory_contents_before_update.shape


def test_dntm_memory_is_zeros_after_reset():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    with torch.no_grad():
        dntm_memory.address_memory(mock_hidden_state)
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
        dntm_memory.reset_memory_content()
    assert (dntm_memory.memory_contents == 0).all()


def test_dntm_controller_hidden_state_contains_no_nan_values_after_update():
    memory_parameters = _init_dntm_memory_parameters()
    mocked_controller_input = _mock_controller_input()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE, batch_size=BATCH_SIZE)
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=memory_parameters["n_locations"],
        controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_output_size=CONTROLLER_OUTPUT_SIZE,
        batch_size=BATCH_SIZE,
    )
    with torch.no_grad():
        dntm(mocked_controller_input)

    assert not dntm.controller_hidden_state.isnan().any()
