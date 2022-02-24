# from ward import test
import torch

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine, DynamicNeuralTuringMachineMemory


# @test("Test DNTM")
def test_dntm_memory_content_shape():
    n_locations = 100
    content_size = 100
    address_size = 20
    example_hidden_state = torch.randn((n_locations, 1))
    dntm_memory = DynamicNeuralTuringMachineMemory(n_locations=n_locations, content_size=content_size,
                                                   address_size=address_size, controller_input_size=100)
    dntm_memory.address_memory(example_hidden_state)
    memory_contents = dntm_memory.read()
    assert memory_contents.shape == (content_size+address_size, 1)
