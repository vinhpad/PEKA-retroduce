
# file structure:read the configs in the specified path
import sys
import os
notebook_path = os.getcwd()
print(notebook_path)
root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
print(f"⭐️ root folder for project: {root}")
code_root = root + "/PEKA/"
env_file_path = f"{code_root}/.env"
data_root = code_root + "/DATA/"
ckpt_folder = f"{root}/Pretrained/"
output_dir = f"{root}/OUTPUT/"
all_configs_path = f"{code_root}/hydra_zen/Configs/"
print(f"😁 code root folder for project: {code_root}")
print(f"💾 data root folder for project: {data_root}")

print(f"🤗 ckpt folder for project: {ckpt_folder}")
print(f"🐯 output folder for project: {output_dir}")
print(f"📃📃 configs root folder for project: {all_configs_path}")

sys.path.append(root)
sys.path.append(f'/{code_root}/')
sys.path.append(f'/{code_root}/peka/External_models/')
sys.path.append(f'/{code_root}/peka/External_models/HEST/src/')

import peka
import dotenv
dotenv.load_dotenv(env_file_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
HF_TOKEN = os.getenv("HF_TOKEN")

from huggingface_hub import login
# Login to the Hugging Face hub, using your user access token that can be found here:
# https://huggingface.co/settings/tokens.
login(token=HF_TOKEN)


# specify the dataset name and path
tissue_type = "breast" # or "other_cancer"
database_name = "breast_visium_26k" # breast_xenium_100k etc.
model_name = "scFoundation"

#   load configs 
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from hydra_zen import builds, instantiate, make_config, zen
from hydra_zen import load_from_yaml

print("🚀 Loading Zen Configs for default settings...")
dataset_config_loc = f"{all_configs_path}/Datasets/{database_name}_{model_name}.yaml"
dataset_config_zen = load_from_yaml(dataset_config_loc)

image_model_name = "H-optimus-0_LoRA_MLP"
model_config_loc = f"{all_configs_path}/Models/{image_model_name}.yaml"
model_config_zen = load_from_yaml(model_config_loc)

optimizers_config_loc = f"{all_configs_path}/Optimizers/default.yaml"
optimizers_config_zen = load_from_yaml(optimizers_config_loc)

trainer_config_loc = f"{all_configs_path}/Trainers/default.yaml"
trainer_config_zen = load_from_yaml(trainer_config_loc)

print("🚀 Instantiating Trainer...")
# instantiate trainer
trainer_instance = instantiate(trainer_config_zen,
                                # Basic configurations
        
        entity=WANDB_ENTITY,
        exp_name=f"PEKA_{database_name}_{model_name}",
        task_type="regression",

        class_nb=1,  # 将类型声明为 Optional[int]

        # 模型
        model_name=image_model_name,
        ckpt_folder=ckpt_folder,

        # Model and training components
        # 最大 epoch 数
        max_epochs=20,
        # 训练输出目录
        trainer_output_dir=output_dir,
        additional_pl_paras={}, 
        # 日志
        # wandb 密钥
        wandb_api_key=WANDB_API_KEY,
                               )

# instantiate dataset, model, and optimizer
print("🚀 Instantiating dataset, model, and optimizer...")
train_loader,val_loader,target_dim = instantiate(dataset_config_zen, data_root=data_root)
model = instantiate(model_config_zen, target_dim=target_dim)
optimizer_instance_list, scheduler_instance_list, metrics_factory, loss_instance = instantiate(optimizers_config_zen)
# instantiate model
from peka.Trainer.pl_basic import pl_basic
pl_model = pl_basic(
    model_instance=model,
    loss_instance=loss_instance(),
    optimizer_instance_list=optimizer_instance_list,
    metrics_factory=metrics_factory,
)


trainer_instance.fit(pl_model, train_loader, val_loader)