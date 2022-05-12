"""This script trains a DNTM on the Deepmind dataset of mathematical problems."""
import hydra
import wandb
import omegaconf
import logging

import torch
from torchmetrics.classification import Accuracy
from torchmetrics import CharErrorRate, MatchErrorRate

from src.utils import configure_reproducibility
from src.data.math_dm import get_dataloader
from src.models.train_dntm_utils import build_model
from src.wandb_utils import log_weights_gradient


@hydra.main(config_path="../../conf", config_name="maths")
def click_wrapper(cfg):
    train_and_test_dntm_maths(cfg)


def train_and_test_dntm_maths(cfg):
    device = torch.device("cuda", 0)
    rng = configure_reproducibility(cfg.run.seed)

    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="dntm_math", entity="flapetr")
    wandb.run.name = cfg.run.codename
    for subconfig_name, subconfig_values in cfg_dict.items():
        if isinstance(subconfig_values, dict):
            wandb.config.update(subconfig_values)
        else:
            logging.warning(f"{subconfig_name} is not being logged.")

    text_table = wandb.Table(columns=["prediction", "target"])

    data_loader, vocab = get_dataloader(cfg)
    model = build_model(cfg.model, device)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=6e-4, betas=(0.99, 0.995), eps=1e-9)

    for epoch in range(cfg.train.epochs):
        train_loss, train_accuracy, cer, mer = train_step(data_loader, vocab, model, criterion, optimizer, device, text_table, cfg)
        print(f"Epoch {epoch} --- train loss: {train_loss} - train acc: {train_accuracy}")

        # valid_step(data_loader)
        wandb.log({'loss_training_set': train_loss})
        wandb.log({'acc_training_set': train_accuracy})
        wandb.log({"char_error_rate_training_set": cer})
        wandb.log({"match_error_rate_training_set": mer})
        log_weights_gradient(model)
    wandb.log({'predictions': text_table})


def train_step(data_loader, vocab, model, criterion, optimizer, device, text_table, cfg):
    train_accuracy = Accuracy().to(device)
    char_error_rate = CharErrorRate().to(device)
    match_error_rate = MatchErrorRate().to(device)

    epoch_loss = 0
    num_batches = 0
    model.train()
    for batch, targets in data_loader:
        num_batches += 1
        batch_size = len(batch)
        target_seq_len = targets.shape[1]  # targets.shape = (bs, target_seq_len)
        batch, targets = batch.to(device), targets.to(device)
        one_hot_batch = torch.nn.functional.one_hot(batch.type(torch.int64), len(vocab)).type(torch.float32)
        one_hot_targets = torch.nn.functional.one_hot(targets.type(torch.int64), len(vocab)).type(torch.float32)
        hn_cn, output = model(one_hot_batch)
        # print(f"{batch.shape=}")
        # print(f"{one_hot_batch.shape=}")
        # print(f"{output.shape=}")
        # print(f"{targets.shape=}")
        # print(f"{one_hot_targets.shape=}")

        current_output = output
        
        current_outputs_str = []
        for output_i, output in enumerate(current_output.T):
           current_outputs_str.append(vocab.lookup_token(output.argmax()))
        
        batch_loss = criterion(current_output.T, targets[:, 0].type(torch.int64))
        # print("generating output")
        # print(f"{current_output.T.shape=}")
        # print(f"{batch_loss=}")
        for target_seq_pos in range(1, target_seq_len):
            target_seq_el = targets[:, target_seq_pos].type(torch.int64)  # target_seq_el.shape = (bs, 1)
            target_seq_el_1hot = one_hot_targets[:, target_seq_pos, :]
            new_input = target_seq_el_1hot.unsqueeze(dim=1)
            hn_cn, current_output = model(new_input)
            batch_loss += criterion(current_output.T, target_seq_el)
            train_accuracy.update(current_output.T, target_seq_el)
            
            for output_i, output in enumerate(current_output.T):
                current_outputs_str[output_i] += vocab.lookup_token(output.argmax())
    
        current_targets_str = []
        for target in targets:
            current_targets_str.append(''.join([vocab.lookup_token(t) for t in target]))
            
        #print(current_targets_str, current_outputs_str)
        text_table.add_data(current_outputs_str[0], current_targets_str[0])
        char_error_rate.update(current_outputs_str, current_targets_str)
        match_error_rate.update(current_outputs_str, current_targets_str)

        # print(f"{target_seq_el=}")
        # print(f"{new_input.shape=}")
        # print(f"{current_output.T.shape=}")
        # print(f"{batch_loss=}")
        # print(f"{current_output.T=}")
        # print(f"{new_input=}")

        # print()
        epoch_loss += batch_loss.item() * batch_size
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm, norm_type=2.0, error_if_nonfinite=True)
        optimizer.step()
    accuracy_over_batches = train_accuracy.compute()
    cer_over_batches = char_error_rate.compute()
    mer_over_batches = match_error_rate.compute()
    epoch_loss /= num_batches
    train_accuracy.reset()
    return epoch_loss, accuracy_over_batches, cer_over_batches, mer_over_batches


def valid_step(data_loader):
    return


if __name__ == "__main__":
    click_wrapper()
