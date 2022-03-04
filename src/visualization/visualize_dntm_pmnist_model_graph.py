"""This script only creates a visualization of the PMNIST DNTM on Tensorboard."""
import torch
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from src.models.DynamicNeuralTuringMachine import DynamicNeuralTuringMachine, DynamicNeuralTuringMachineMemory

timestamp = (str(datetime.utcnow())
             .replace(' ', '-')
             .replace('.', '-')
             .replace(':', '-'))
writer = SummaryWriter(log_dir=f'runs/visualize_dntm_pmnist_model_graph/{timestamp}')

n_locations = 12
controller_input_size = 1
controller_output_size = 10
dntm_memory = DynamicNeuralTuringMachineMemory(
    n_locations=n_locations,
    content_size=8,
    address_size=2,
    controller_input_size=controller_input_size
)
dntm = DynamicNeuralTuringMachine(
    memory=dntm_memory,
    controller_hidden_state_size=n_locations,
    controller_input_size=controller_input_size,
    controller_output_size=controller_output_size
).to("cuda")

mocked_input = torch.ones(size=(1, 1), device="cuda")

dntm(mocked_input)

writer.add_graph(dntm, mocked_input)
writer.close()
