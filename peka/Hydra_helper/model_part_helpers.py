"""
some naive code to simplify the usage of hydra-zen

"""
import torch
import timm
import peft

# method name -> peft config class name, resolved lazily so that an older peft only
# breaks when you actually select a method it does not ship
_PEFT_CONFIG_CLASSES = {
    "lora": ("LoraConfig", "0.3.0"),
    "adalora": ("AdaLoraConfig", "0.3.0"),
    "hra": ("HRAConfig", "0.12.0"),
    "bone": ("BoneConfig", "0.14.0"),
}


def _resolve_peft_config_cls(peft_method: str):
    if peft_method not in _PEFT_CONFIG_CLASSES:
        raise ValueError(
            f"Unsupported peft_method: {peft_method}. "
            f"Supported: {sorted(_PEFT_CONFIG_CLASSES)}"
        )
    cls_name, since = _PEFT_CONFIG_CLASSES[peft_method]
    cls = getattr(peft, cls_name, None)
    if cls is None:
        raise ImportError(
            f"peft_method '{peft_method}' needs {cls_name}, which is not available in "
            f"peft {getattr(peft, '__version__', '?')} (added in peft {since}). "
            f"Upgrade with:  pip install -U 'peft>={since}'"
        )
    return cls
from omegaconf import ListConfig
from typing import Optional

from peka import logger
from peka.Model.utils import get_module
from peka.Model.base import HistoPath_AlignmentModel

def model_config(
                        encoder_name, 
                        encoder_output_dim, 
                        translate_module_name, # check src/Models/utils.py get_module()
                        target_dim, # target dimension,
                                        
                        lora_r, # LoRA rank
                        lora_alpha, # LoRA alpha
                         lora_dropout, # LoRA dropout

                         pre_trained_ckpt_path:str=None,
                         translate_additional_params:Optional[ListConfig]=None, # 传递给translate_module_name的额外参数
                         ):

    encoder = timm.create_model(
                            #"hf-hub:bioptimus/H-optimus-0", 
                            encoder_name, 
                            pretrained=True, init_values=1e-5, dynamic_img_size=False
                        )
    # 模型参数

    peft_method = translate_additional_params.get("peft_method", "lora") if translate_additional_params else "lora"
    
    peft_config_cls = _resolve_peft_config_cls(peft_method)

    if peft_method == "lora":
        peft_config = peft_config_cls(
            r=lora_r,  # LoRA rank, dimension of low-rank matrix
            lora_alpha=lora_alpha,  # scale factor, controls the impact of LoRA
            lora_dropout=lora_dropout,  # dropout rate of LoRA layer, prevents overfitting
            target_modules=None,  # target modules, specify which layers apply LoRA, None means automatic selection
        )
    elif peft_method == "adalora":
        peft_config = peft_config_cls(
            r=lora_r,  # initial LoRA rank
            lora_alpha=lora_alpha,  # scale factor
            lora_dropout=lora_dropout,  # dropout rate
            target_modules=None,  # target modules
            target_r=translate_additional_params.get("target_r", 8),  # target average rank, AdaLoRA will dynamically adjust to this value
            init_r=translate_additional_params.get("init_r", 12),  # initial rank of each incremental matrix
            beta1=translate_additional_params.get("beta1", 0.85),  # EMA hyperparameter for smoothing sensitivity
            beta2=translate_additional_params.get("beta2", 0.85),  # EMA hyperparameter for uncertainty quantification
            total_step=translate_additional_params.get("total_step", None),  # total training steps, used for budget allocation
        )
    elif peft_method == "hra":
        peft_config = peft_config_cls(
            r=lora_r,  # HRA rank, suggest to set as even number for default initialization method to work
            apply_GS=translate_additional_params.get("apply_GS", False),  # whether apply Gram-Schmidt orthogonization
            target_modules=None,  # target modules, specify which layers apply HRA
        )
    elif peft_method == "bone":
        peft_config = peft_config_cls(
            r=lora_r,  # Bone rank, suggest to set as even number for default initialization method to work
            target_modules=None,  # target modules
            init_weights=translate_additional_params.get("init_weights", True),  # weight initialization method, True uses Bone structure, 'bat' uses Bat structure
        )

    translate_model = get_module(translate_module_name,
                                encoder_output_dim, 
                                target_dim,
                                additional_params=translate_additional_params)

    # instance backbone
    model = HistoPath_AlignmentModel(encoder=encoder, translate_model=translate_model, lora_paras=peft_config)


    if pre_trained_ckpt_path is not None:
        logger.info(f"loading pre-trained ckpt from {pre_trained_ckpt_path}")
        model.load_state_dict(torch.load(pre_trained_ckpt_path))
    else:
        logger.info(f"no pre-trained ckpt provided, using random initialization")

    logger.info(f"model initialized with {encoder_name} encoder and {translate_module_name} translate model")

    return model