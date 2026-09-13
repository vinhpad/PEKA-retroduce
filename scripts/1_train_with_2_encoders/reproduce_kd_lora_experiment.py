"""
Script to reproduce an experiment using saved configurations.
"""
import sys
import os
import argparse
from datetime import datetime

# Setup paths
notebook_path = os.getcwd()
root = os.path.dirname(os.path.dirname(os.path.dirname(notebook_path)))
code_root = root + "/PEKA/"
env_file_path = f"{code_root}/.env"
data_root = code_root + "DATA/"
ckpt_folder = f"{root}/Pretrained/"
output_dir = f"{root}/OUTPUT/"

sys.path.append(root)
sys.path.append(f'/{code_root}/')
sys.path.append(f'/{code_root}/peka/External_models/')
sys.path.append(f'/{code_root}/peka/External_models/HEST/src/')

import peka
import dotenv
dotenv.load_dotenv(env_file_path)
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")

from hydra_zen import instantiate
from peka.Trainer.KD_LoRA import pl_KD_LoRA
from peka.Hydra_helper.pl_model_helpers import create_pl_model
from peka.Hydra_helper.experiment_helpers import load_experiment_configs, find_latest_experiment

def parse_args():
    parser = argparse.ArgumentParser(description='Reproduce a saved experiment')
    parser.add_argument('--experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('--latest', action='store_true', help='Use latest experiment')
    parser.add_argument('--prefix', type=str, help='Prefix to filter experiments when using --latest')
    parser.add_argument('--phase1_epochs', type=int, default=20, help='Number of epochs for phase 1 training')
    parser.add_argument('--phase1_lr', type=float, default=1e-4, help='Learning rate for phase 1 training')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name for wandb logging')
    return parser.parse_args()

def main():
    args = parse_args()

    # Find experiment directory
    if args.latest:
        experiment_dir = find_latest_experiment(output_dir, args.prefix)
    else:
        if args.experiment_dir is None:
            raise ValueError("Must provide either --experiment_dir or --latest")
        experiment_dir = args.experiment_dir

    # Load configurations
    configs = load_experiment_configs(experiment_dir)
    for required in ("dataset", "model", "optimizer", "trainer", "pl_model"):
        if required not in configs:
            raise ValueError(f"Saved experiment is missing the '{required}' config: {experiment_dir}")
    pl_model_config = configs['pl_model']

    # Create new experiment directory for reproduction
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reproduction_dir = os.path.join(output_dir, f"reproduction_{os.path.basename(experiment_dir)}_{timestamp}")
    os.makedirs(reproduction_dir)
    exp_name = args.exp_name or f"reproduce_{os.path.basename(experiment_dir)}"

    USE_SPLIT_DATASET = True

    # Load dataset
    train_loader, val_loader, target_dim = instantiate(
        configs['dataset'],
        data_root=data_root,
        split_dataset=USE_SPLIT_DATASET,
        val_ratio=0.2,
        split_seed=42,
        random_sample_barcode=True if not USE_SPLIT_DATASET else False
    )

    # Phase 1: Train MLP classifier
    teacher_classifier = pl_KD_LoRA.train_phase1(
        train_loader=train_loader,
        val_loader=val_loader,
        input_dim=target_dim,
        classifier_hidden_dim=pl_model_config.classifier_hidden_dim,
        num_classes=pl_model_config.num_classes,
        save_path=os.path.join(reproduction_dir, "phase1", "classifier.pt"),
        device='cuda',
        num_epochs=args.phase1_epochs,
        learning_rate=args.phase1_lr
    )

    # Initialize student model with LoRA
    model = instantiate(configs['model'], target_dim=target_dim)

    # Rebuild the optimizer/scheduler/metrics/loss stack from the saved optimizer config
    optimizer_instance_list, scheduler_instance_list, metrics_factory, loss_instance = \
        instantiate(configs['optimizer'])

    # Phase 2: Train LoRA model with knowledge distillation
    pl_model_config.lora_save_path = os.path.join(reproduction_dir, "phase2", "lora")
    kd_model = create_pl_model(
        model_instance=model,
        optimizer_instance_list=optimizer_instance_list,
        scheduler_instance_list=scheduler_instance_list,
        metrics_factory=metrics_factory,
        loss_instance=loss_instance,
        pl_model_config=pl_model_config,
        model_type="kd_lora"
    )

    # Setup teacher model
    kd_model.setup_teacher_model(teacher_classifier)

    # Rebuild the trainer from the saved trainer config so epochs, callbacks and
    # logging match the original run
    trainer = instantiate(
        configs['trainer'],
        entity=WANDB_ENTITY,
        exp_name=exp_name,
        task_type="classification",
        class_nb=pl_model_config.num_classes,
        model_name=os.path.basename(experiment_dir),
        ckpt_folder=ckpt_folder,
        trainer_output_dir=os.path.join(reproduction_dir, "phase2"),
        wandb_api_key=WANDB_API_KEY,
    )

    # Train model
    trainer.fit(kd_model, train_loader, val_loader)

    print(f"✨ Experiment reproduced in: {reproduction_dir}")

if __name__ == "__main__":
    main()
