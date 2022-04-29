import torch

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine
from src.models.DynamicNeuralTuringMachineMemory import DynamicNeuralTuringMachineMemory


CONTROLLER_INPUT_SIZE = 1
CONTROLLER_HIDDEN_STATE_SIZE = 13
CONTROLLER_OUTPUT_SIZE = 10
BATCH_SIZE = 4


def _init_dntm_memory_parameters():
    n_locations = 7
    content_size = 6
    address_size = 2
    return {
        "n_locations": n_locations,
        "content_size": content_size,
        "address_size": address_size,
    }


def _mock_controller_input():
    return torch.ones((CONTROLLER_INPUT_SIZE, BATCH_SIZE))


def _mock_controller_hidden_state():
    return torch.randn((CONTROLLER_HIDDEN_STATE_SIZE, BATCH_SIZE))


def test_dntm_memory_reading_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    with torch.no_grad():
        dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
        memory_reading = dntm_memory.read(mock_hidden_state)
    assert memory_reading.shape == (
        memory_parameters["content_size"] + memory_parameters["address_size"], BATCH_SIZE)


def test_dntm_memory_address_vector_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    with torch.no_grad():
        dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
        address_vector = dntm_memory._address_memory(mock_hidden_state)
    assert address_vector.shape == (
        memory_parameters["n_locations"], BATCH_SIZE)


def test_dntm_memory_address_vector_contains_no_nan_values():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()
    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    with torch.no_grad():
        dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
        address_vector = dntm_memory._address_memory(mock_hidden_state)
    assert not address_vector.isnan().any()


# this test is no longer meaningful with the new implementation of the NO-OP
# def test_dntm_memory_address_vector_sum_to_one():
#     memory_parameters = _init_dntm_memory_parameters()
#     mock_hidden_state = _mock_controller_hidden_state()
#     dntm_memory = DynamicNeuralTuringMachineMemory(
#         **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE)
#     with torch.no_grad():
#         dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
#         address_vector = dntm_memory._address_memory(mock_hidden_state)
#     assert address_vector.sum().item() == pytest.approx(BATCH_SIZE)


def test_dntm_memory_contents_shape_doesnt_change_after_update():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    with torch.no_grad():
        dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
        memory_contents_before_update = dntm_memory.memory_contents
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
    assert dntm_memory.memory_contents.shape == memory_contents_before_update.shape


def test_dntm_memory_is_zeros_after_reset():
    memory_parameters = _init_dntm_memory_parameters()
    mock_hidden_state = _mock_controller_hidden_state()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    with torch.no_grad():
        dntm_memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))
        dntm_memory.update(mock_hidden_state, _mock_controller_input())
        dntm_memory.reset_memory_content()
    assert (dntm_memory.memory_contents == 0).all()


def test_dntm_controller_hidden_state_contains_no_nan_values_after_update():
    memory_parameters = _init_dntm_memory_parameters()
    mocked_controller_input = _mock_controller_input()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE,
        controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_output_size=CONTROLLER_OUTPUT_SIZE,
    )

    dntm.reshape_and_reset_hidden_states(batch_size=BATCH_SIZE, device=torch.device("cpu"))
    dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))

    with torch.no_grad():
        dntm(mocked_controller_input)

    assert not dntm.controller_hidden_state.isnan().any()


def test_dntm_output_shape():
    memory_parameters = _init_dntm_memory_parameters()
    mocked_controller_input = _mock_controller_input()

    dntm_memory = DynamicNeuralTuringMachineMemory(
        **memory_parameters, controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE)
    dntm = DynamicNeuralTuringMachine(
        memory=dntm_memory,
        controller_hidden_state_size=CONTROLLER_HIDDEN_STATE_SIZE,
        controller_input_size=CONTROLLER_INPUT_SIZE,
        controller_output_size=CONTROLLER_OUTPUT_SIZE,
    )

    dntm.reshape_and_reset_hidden_states(batch_size=BATCH_SIZE, device=torch.device("cpu"))
    dntm.memory.reshape_and_reset_exp_mov_avg_sim(batch_size=BATCH_SIZE, device=torch.device("cpu"))

    with torch.no_grad():
        _, output = dntm(mocked_controller_input)

    assert output.shape == (10, 4)
