import logging


def log_weights_gradient(model, wandb):
    for param_name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log({f"{param_name}_gradient": wandb.Histogram(param.grad.cpu())})
        else:
            logging.warning(f"{param_name} gradient is None!")
