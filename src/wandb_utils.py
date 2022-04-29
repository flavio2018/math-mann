import logging
import wandb


def log_weights_gradient(model):
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"{param_name}_gradient": wandb.Histogram(param.grad.cpu())})
        else:
            logging.warning(f"{param_name} gradient is None!")


def log_preds_and_targets(batch_i, output, targets):
    if batch_i == 0:
        columns = ["Predictions", "Targets"]
        data = zip([str(p.item()) for p in output.argmax(axis=0)],
                   [str(t.item()) for t in targets])
        data = [list(row) for row in data]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({"First batch preds vs targets": table})
