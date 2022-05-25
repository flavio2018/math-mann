import hydra
import omegaconf

from src.utils import configure_reproducibility
from src.data.math_dm import get_dataloaders


@hydra.main("../../conf", "test_data_math")
def main(cfg):
	rng = configure_reproducibility(cfg.run.seed)
	cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

	train_dataloader, valid_dataloader, vocab = get_dataloaders(cfg, rng)
	visualize_vocab(vocab)
	visualize_first_element_string(train_dataloader, vocab)
	visualize_first_element_string(valid_dataloader, vocab)
	visualize_first_element(train_dataloader)
	visualize_first_element(train_dataloader)
	

def get_sequence_strings(batch, vocab):
	batch_strings = []
	for sequence in batch:
		batch_strings.append(''.join([vocab.lookup_token(t) for t in sequence]))
	return batch_strings


def visualize_first_element_string(data_loader, vocab):
	print_separator()
	for batch, targets, _ in data_loader:
		print_batch_targets_stats(batch, targets)
		inputs_strings = get_sequence_strings(batch, vocab)
		targets_strings = get_sequence_strings(targets, vocab)
		for input_str, target_str in zip(inputs_strings, targets_strings):
			print(input_str, target_str, sep='')
		return


def visualize_first_element(data_loader):
	print_separator()
	for batch, targets, masks in data_loader:
		masks_X, masks_Y = masks
		print_batch_targets_stats(batch, targets)
		print(batch[0])
		print(masks_X[0])
		print(targets[0])
		print(masks_Y[0])
		return


def visualize_vocab(vocab):
	print_separator()
	print("Token mapping to index:", vocab.get_stoi())
	print("Token:", vocab.get_itos())
	print("Default index:", vocab.get_default_index())



def print_batch_targets_stats(batch, targets):
	batch_size, sequence_len = batch.shape
	max_targets_len = targets.shape[1]
	print(f"Batch of size {batch_size}, max sequence lenght {sequence_len} and max targets length {max_targets_len}")


def print_separator():
	print('\n' + 40*'-')
	

if __name__ == '__main__':
	main()
