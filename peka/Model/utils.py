import torch.nn as nn
from omegaconf import ListConfig
from typing import Optional
def get_module(module_name, input_dim, output_dim, additional_params:Optional[ListConfig]=None):
    # check if additional_params is ListConfig and convert to dict
    if isinstance(additional_params, ListConfig):
        additional_params = {k: v for d in additional_params for k, v in d.items()}


    if module_name == 'MLP':
        # additional_params is None for the default configs, so mid_dim needs a
        # fallback here rather than being left unbound
        mid_dim = 512
        if additional_params is not None:
            mid_dim = additional_params.get('mid_dim', mid_dim)
        return nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, output_dim)
        )
    elif module_name == 'Transformer':
        # a transformer module; peft_method belongs to the PEFT config, not to nn.Transformer
        transformer_params = dict(additional_params) if additional_params is not None else {}
        transformer_params.pop('peft_method', None)
        return nn.Sequential(
                            nn.Transformer(
                                d_model=input_dim,
                                **transformer_params
                        ),
                        nn.Linear(input_dim, output_dim)
                        )
    else:
        raise ValueError(f"Unknown module name: {module_name}")


def get_loss_function(loss_name):
    if loss_name == 'MSE':
        return nn.MSELoss
    elif loss_name == 'KLDiv':
        return nn.KLDivLoss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    