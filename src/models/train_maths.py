"""This script trains a DNTM on the Deepmind dataset of mathematical problems."""
import hydra
import wandb
import omegaconf
import logging
import os

import numpy as np
import torch
from torchmetrics.classification import Accuracy
from torchmetrics import CharErrorRate, MatchErrorRate

from src.utils import configure_reproducibility
from src.data.math_dm import get_dataloaders
from src.models.train_dntm_utils import build_model
from src.wandb_utils import log_weights_gradient
from src.models.pytorchtools import EarlyStopping


@hydra.main(config_path="../../conf", config_name="maths_slurm")
def click_wrapper(cfg):
    train_and_test_dntm_maths(cfg)


def train_and_test_dntm_maths(cfg):
    device = torch.device("cuda", 0)
    rng = configure_reproducibility(cfg.run.seed)

    cfg_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project="dntm_math", entity="flapetr", mode=cfg.run.wandb_mode)
    wandb.run.name = cfg.run.codename
    for subconfig_name, subconfig_values in cfg_dict.items():
        if isinstance(subconfig_values, dict):
            wandb.config.update(subconfig_values)
        else:
            logging.warning(f"{subconfig_name} is not being logged.")

    text_table = wandb.Table(columns=["epoch", "prediction", "target"])

    train_dataloader, valid_dataloader, vocab = get_dataloaders(cfg, rng)

    model = build_model(cfg.model, device)

    criterion = torch.nn.NLLLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, betas=(0.99, 0.995), eps=1e-9)
    early_stopping = EarlyStopping(verbose=True,
                                   path=os.path.join(os.getcwd(),
                                                     f"{cfg.run.codename}.pth"),
                                   trace_func=logging.info,
                                   patience=cfg.train.patience)

    for epoch in range(cfg.train.epochs):
        train_loss, train_accuracy, cer, mer = step_on_dataset(train_dataloader, model, criterion, optimizer,
                                                               device, vocab, text_table, cfg, epoch, train=True)
        valid_loss, valid_accuracy, valid_cer, valid_mer = step_on_dataset(train_dataloader, model, criterion, optimizer,
                                                                           device, vocab, text_table, cfg, epoch, train=False)

        print(f"Epoch {epoch} --- train loss: {train_loss} - train acc: {train_accuracy} - "
              f"valid loss: {valid_loss} - valid acc: {valid_accuracy}")
        wandb.log({'loss_training_set': train_loss})
        wandb.log({'loss_validation_set': valid_loss})
        wandb.log({'acc_training_set': train_accuracy})
        wandb.log({'acc_validation_set': valid_accuracy})
        wandb.log({"char_error_rate_training_set": cer})
        wandb.log({"match_error_rate_training_set": mer})
        log_weights_gradient(model)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

    wandb.log({'predictions': text_table})


def step_on_dataset(dataloader, model, criterion, optimizer,
    device, vocab, text_table, cfg, epoch, train: bool):
    char_error_rate = CharErrorRate().to(device)
    match_error_rate = MatchErrorRate().to(device)
    global_accuracy = 0
    epoch_loss = 0
    num_samples = 0
    num_batches = 0
    
    if train:
        model.train()
    else:
        model.eval()

    for batch, targets, masks in dataloader:
        num_batches += 1
        batch_size = len(batch)
        masks_X, masks_Y = masks
        input_sequences_lengths = masks_X.sum(axis=1).to(device)
        target_sequences_lengths = masks_Y.sum(axis=1).to(device)
        target_seq_len = targets.shape[1]  # targets.shape = (bs, target_seq_len)
        batch, targets = batch.to(device), targets.to(device)
        one_hot_batch = torch.nn.functional.one_hot(batch.type(torch.int64), len(vocab)).type(torch.float32)
        one_hot_targets = torch.nn.functional.one_hot(targets.type(torch.int64), len(vocab)).type(torch.float32)
        
        if train:
            model.zero_grad()

        model.prepare_for_batch(one_hot_batch, device)

        hidden_states, outputs = model(one_hot_batch)
        output = torch.stack([outputs[l-1,:,b] for l, b in zip(input_sequences_lengths, range(batch_size))]) 
        
        hidden_state = model.set_hidden_state(hidden_states, input_sequences_lengths, batch_size)
        current_output = output
        predictions = [current_output.T]
        
        current_outputs_str = []
        current_outputs_str = update_outputs_str(current_outputs_str, current_output, vocab)
        
        for target_seq_pos in range(1, target_seq_len):
            target_seq_el = targets[:, target_seq_pos].type(torch.int64)  # target_seq_el.shape = (bs, 1)
            target_seq_el_1hot = one_hot_targets[:, target_seq_pos, :]
            new_input = target_seq_el_1hot.unsqueeze(dim=1)
            hn_cn, current_output = model(new_input)
            current_output = current_output.reshape(current_output.shape[1:])
            predictions.append(current_output)
            current_outputs_str = update_outputs_str(current_outputs_str, current_output, vocab)
        
        predictions = torch.stack(predictions).reshape(batch_size, len(vocab), target_seq_len)
        batch_loss_2d = criterion(predictions, targets.type(torch.int64))
        batch_loss_2d = torch.where(masks_Y.to(device), batch_loss_2d, torch.zeros_like(batch_loss_2d))
        batch_loss = torch.mean(batch_loss_2d.sum(axis=1) / target_sequences_lengths)
        
        current_targets_str = []
        for target in targets:
            current_targets_str.append(''.join([vocab.lookup_token(t) for t in target]))
            
        if train and (epoch % 3 == 0) and (num_batches % 2 == 0):
            text_table.add_data(epoch, current_outputs_str[0], current_targets_str[0])
        correct_in_batch, num_samples_in_batch = num_correct_in_batch(current_outputs_str, current_targets_str)
        global_accuracy += correct_in_batch
        num_samples += num_samples_in_batch
        char_error_rate.update(current_outputs_str, current_targets_str)
        match_error_rate.update(current_outputs_str, current_targets_str)
        epoch_loss += batch_loss.item() * batch_size

        if train:
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm, norm_type=2.0, error_if_nonfinite=True)
            optimizer.step()
    global_accuracy /= num_samples
    cer_over_batches = char_error_rate.compute()
    mer_over_batches = match_error_rate.compute()
    epoch_loss /= num_batches
    return epoch_loss, global_accuracy, cer_over_batches, mer_over_batches


def update_outputs_str(outputs_str, current_output, vocab):
    if len(outputs_str) == 0:
        for output in current_output:
            outputs_str.append(vocab.lookup_token(output.argmax()))
    else:
        for output_i, output in enumerate(current_output.T):
            outputs_str[output_i] += vocab.lookup_token(output.argmax())
    return outputs_str


def num_correct_in_batch(batch_preds, batch_targets):
    batch_preds, batch_targets = np.array(batch_preds), np.array(batch_targets)
    num_correct_in_batch = (batch_preds == batch_targets).sum()
    return num_correct_in_batch, len(batch_preds)


if __name__ == "__main__":
    click_wrapper()
