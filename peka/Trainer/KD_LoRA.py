import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from peft import LoraConfig, get_peft_model
import os
import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam

from peka.Model.base import MLPClassifier
from peka.Trainer.pl_basic import pl_basic

class pl_KD_LoRA(pl_basic):
    """
    A class for training a student model using knowledge distillation with LoRA.
    """
    def __init__(
        self,
        model_instance: nn.Module,  # student model (HistoPath_AlignmentModel)
        loss_instance,
        metrics_factory,
        optimizer_instance_list: list,
        scheduler_instance_list: list = None,
        num_classes: int = None,
        classifier_hidden_dim: int = 512,
        input_dim: int = 1536,  # scLLM embedding dimension
        temperature: float = 2.0,
        alpha: float = 0.5,  # weight for distillation loss
        lora_save_path: str = None,
        prototype_loss_weight: float = 0.0,
        prototype_temperature: float = 0.1,
        prototype_warmup_epochs: int = 5,
        prototype_ema_momentum: float = 0.95,
        teacher_anchor_weight: float = 0.5,
    ):
        super().__init__(
            model_instance=model_instance,
            loss_instance=loss_instance,
            metrics_factory=metrics_factory,
            optimizer_instance_list=optimizer_instance_list,
            scheduler_instance_list=scheduler_instance_list,
        )
        
        self.temperature = temperature
        self.alpha = alpha
        self.lora_save_path = lora_save_path
        self.input_dim = input_dim
        self.classifier_hidden_dim = classifier_hidden_dim
        self.num_classes = num_classes
        self.prototype_loss_weight = prototype_loss_weight
        self.prototype_temperature = prototype_temperature
        self.prototype_warmup_epochs = prototype_warmup_epochs
        self.prototype_ema_momentum = prototype_ema_momentum
        self.teacher_anchor_weight = teacher_anchor_weight

        if self.prototype_temperature <= 0:
            raise ValueError("prototype_temperature must be positive")
        if self.prototype_loss_weight < 0:
            raise ValueError("prototype_loss_weight must be non-negative")
        if self.prototype_warmup_epochs < 0:
            raise ValueError("prototype_warmup_epochs must be non-negative")
        if not 0 <= self.prototype_ema_momentum <= 1:
            raise ValueError("prototype_ema_momentum must be in [0, 1]")
        if not 0 <= self.teacher_anchor_weight <= 1:
            raise ValueError("teacher_anchor_weight must be in [0, 1]")

        # Rebuilt from the training split for each run; inference and older checkpoints
        # must not depend on these training-only buffers.
        self.register_buffer("teacher_prototypes", torch.empty(0), persistent=False)
        self.register_buffer("dynamic_prototypes", torch.empty(0), persistent=False)
        self.register_buffer("prototype_epoch_sums", torch.empty(0), persistent=False)
        self.register_buffer("prototype_epoch_counts", torch.empty(0), persistent=False)
        
        # Initialize classifier for phase 2
        self.classifier = None

    @staticmethod
    def prepare_training_clusters(train_loader: DataLoader, val_loader: DataLoader,
                                  num_classes: int, diagnostic_sample_size: int = 2000):
        """Fit clusters on train embeddings and assign validation by train centroids."""
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        if not hasattr(train_dataset, "iter_embedding_batches"):
            raise TypeError("Dataset does not expose teacher embeddings")

        embedding_batches = list(train_dataset.iter_embedding_batches())
        if not embedding_batches:
            raise ValueError("No teacher embeddings found in the training split")
        embeddings = np.concatenate(embedding_batches)
        normalized_embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
        kmeans = MiniBatchKMeans(
            n_clusters=num_classes,
            random_state=42,
            batch_size=min(4096, len(normalized_embeddings)),
            n_init=10,
        )
        kmeans.fit(normalized_embeddings)
        prototypes = kmeans.cluster_centers_
        prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
        labels = np.argmax(normalized_embeddings @ prototypes.T, axis=1)
        counts = np.bincount(labels, minlength=num_classes)
        missing = np.flatnonzero(counts == 0)
        if len(missing):
            raise ValueError(f"Training split has empty clusters: {missing.tolist()}")

        train_dataset.assign_cluster_labels(prototypes)
        val_dataset.assign_cluster_labels(prototypes)

        diagnostics = {
            "cluster_min_size": int(counts.min()),
            "cluster_max_size": int(counts.max()),
        }
        if diagnostic_sample_size > 0:
            rng = np.random.default_rng(42)
            sample_size = min(diagnostic_sample_size, len(normalized_embeddings))
            sample_indices = rng.choice(len(normalized_embeddings), size=sample_size, replace=False)
            sampled_x = normalized_embeddings[sample_indices]
            sampled_y = labels[sample_indices]
            unique_labels = np.unique(sampled_y)
            if 1 < len(unique_labels) < len(sampled_y):
                diagnostics.update({
                    "cluster_silhouette": float(silhouette_score(sampled_x, sampled_y)),
                    "cluster_calinski_harabasz": float(calinski_harabasz_score(sampled_x, sampled_y)),
                    "cluster_davies_bouldin": float(davies_bouldin_score(sampled_x, sampled_y)),
                })
        return torch.from_numpy(prototypes).float(), diagnostics

    def setup_prototypes(self, teacher_prototypes: torch.Tensor):
        if teacher_prototypes.ndim != 2 or teacher_prototypes.shape[0] != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} two-dimensional prototypes, "
                f"got shape {tuple(teacher_prototypes.shape)}"
            )
        normalized = F.normalize(teacher_prototypes.float(), dim=-1)
        self.teacher_prototypes = normalized
        self.dynamic_prototypes = normalized.clone()
        self.prototype_epoch_sums = torch.zeros_like(normalized)
        self.prototype_epoch_counts = torch.zeros(self.num_classes, dtype=torch.float32)

    def prototype_loss(self, student_features, teacher_features):
        prototypes = self.dynamic_prototypes
        if prototypes.numel() == 0:
            raise RuntimeError("Prototype loss is enabled but setup_prototypes() was not called")
        student_logits = F.normalize(student_features, dim=-1) @ prototypes.T
        teacher_logits = F.normalize(teacher_features, dim=-1) @ prototypes.T
        student_logits = student_logits / self.prototype_temperature
        teacher_probs = F.softmax(teacher_logits / self.prototype_temperature, dim=-1)
        loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction="batchmean")
        accuracy = (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float().mean()
        return loss, accuracy

    def _accumulate_student_prototypes(self, student_features, labels):
        normalized = F.normalize(student_features.detach(), dim=-1)
        self.prototype_epoch_sums.index_add_(0, labels, normalized)
        self.prototype_epoch_counts.index_add_(
            0, labels, torch.ones_like(labels, dtype=self.prototype_epoch_counts.dtype)
        )

    def _update_dynamic_prototypes(self):
        if self.current_epoch + 1 < self.prototype_warmup_epochs:
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.prototype_epoch_sums)
            torch.distributed.all_reduce(self.prototype_epoch_counts)
        present = self.prototype_epoch_counts > 0
        if not torch.any(present):
            return
        student_centroids = self.prototype_epoch_sums[present] / self.prototype_epoch_counts[present, None]
        student_centroids = F.normalize(student_centroids, dim=-1)
        updated = (
            self.prototype_ema_momentum * self.dynamic_prototypes[present]
            + (1 - self.prototype_ema_momentum) * student_centroids
        )
        updated = F.normalize(updated, dim=-1)
        anchored = (
            self.teacher_anchor_weight * self.teacher_prototypes[present]
            + (1 - self.teacher_anchor_weight) * updated
        )
        self.dynamic_prototypes[present] = F.normalize(anchored, dim=-1)

    @staticmethod
    def train_phase1(train_loader: DataLoader, 
                    val_loader: DataLoader, 
                    input_dim: int,
                    classifier_hidden_dim: int,
                    num_classes: int,
                    save_path: str,
                    device: str = 'cuda',
                    num_epochs: int = 20,
                    learning_rate: float = 1e-4) -> MLPClassifier:
        """
        Independent Phase 1 training function, training a MLP classifier
        
        Args:
            train_loader: training data loader, returns (img, emb, label)
            val_loader: validation data loader
            input_dim: input dimension
            classifier_hidden_dim: hidden dimension of the classifier
            num_classes: number of classes
            save_path: path to save the model
            device: device to train on
            num_epochs: number of epochs
            learning_rate: learning rate
            
        Returns:
            Trained MLP classifier
        """
        print(f"🚀 Phase 1: Training MLP Classifier...")
        
        # Initialize classifier
        classifier = MLPClassifier(input_dim, classifier_hidden_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=learning_rate)
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            classifier.train()
            total_loss = 0
            correct = 0
            total = 0
            
            train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            for _, emb, label in train_pbar:
                emb, label = emb.to(device), label.to(device)
                
                optimizer.zero_grad()
                logits = classifier(emb)
                loss = F.cross_entropy(logits, label)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()
                
                train_pbar.set_postfix({
                    'loss': total_loss / (train_pbar.n + 1),
                    'acc': 100. * correct / total
                })
            
            # Validation
            classifier.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
                for _, emb, label in val_pbar:
                    emb, label = emb.to(device), label.to(device)
                    
                    logits = classifier(emb)
                    loss = F.cross_entropy(logits, label)
                    
                    val_loss += loss.item()
                    _, predicted = logits.max(1)
                    val_total += label.size(0)
                    val_correct += predicted.eq(label).sum().item()
                    
                    val_pbar.set_postfix({
                        'loss': val_loss / (val_pbar.n + 1),
                        'acc': 100. * val_correct / val_total
                    })
            
            val_acc = 100. * val_correct / val_total
            print(f'Epoch {epoch+1}: Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = classifier.state_dict()
                print(f'New best model saved! Val Acc: {val_acc:.2f}%')
        
        # Load best model and save
        classifier.load_state_dict(best_model_state)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_model_state, save_path)
        print(f'Best model saved to {save_path} with Val Acc: {best_val_acc:.2f}%')
        
        return classifier

    @staticmethod
    def load_phase1_model(checkpoint_path: str,
                         input_dim: int,
                         classifier_hidden_dim: int,
                         num_classes: int,
                         device: str = 'cuda') -> MLPClassifier:
        """
        Load the trained MLP classifier from checkpoint
        
        Args:
            checkpoint_path: checkpoint path
            input_dim: input dimension
            classifier_hidden_dim: hidden dimension of the classifier
            num_classes: number of classes
            device: device to load on
            
        Returns:
            Loaded MLP classifier
        """
        print(f"🔄 Loading MLP Classifier from {checkpoint_path}")
        
        # Initialize classifier
        classifier = MLPClassifier(input_dim, classifier_hidden_dim, num_classes).to(device)
        
        # Load checkpoint
        classifier.load_state_dict(torch.load(checkpoint_path))
        classifier.eval()  # Set to evaluation mode
        
        return classifier

    def setup_teacher_model(self, classifier: MLPClassifier):
        """Setup teacher model for phase 2"""
        self.classifier = classifier
        self.classifier.requires_grad_(False)  # Freeze classifier during LoRA training

    def distillation_loss(self, student_logits, teacher_logits, labels, with_log=False):
        """Compute knowledge distillation loss"""
        # Compute soft distillation loss with temperature scaling
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)
        
        # Compute hard loss with ground truth labels
        #hard_loss = F.cross_entropy(student_logits, labels)
        hard_loss = self.loss_fn(student_logits, labels)

        if with_log:
            self.log("soft_loss", soft_loss)
            self.log("hard_loss", hard_loss)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss

    def training_step(self, batch, batch_idx):
        """Phase 2: LoRA training with knowledge distillation"""
        # Data preprocess - expect (img, emb, label) from dataloader
        img, emb, label = batch
        
        # Forward pass through student model
        student_features = self.model(img)  # student_features should have the same dimension as scLLM embedding
        student_logits = self.classifier(student_features)
        
        # Teacher logits are from pre-computed embeddings
        with torch.no_grad():
            teacher_logits = self.classifier(emb)
        
        # Compute distillation loss
        loss = self.distillation_loss(student_logits, teacher_logits, label, with_log=True)
        if self.prototype_loss_weight > 0:
            prototype_loss, prototype_accuracy = self.prototype_loss(student_features, emb)
            loss = loss + self.prototype_loss_weight * prototype_loss
            self.log("train_prototype_loss", prototype_loss, on_step=False, on_epoch=True)
            self.log("train_prototype_accuracy", prototype_accuracy, on_step=False, on_epoch=True)
            self._accumulate_student_prototypes(student_features, label)
        
        # Post process and logging
        out = self.train_post_process(student_logits, teacher_logits, loss)
        self.train_epoch_datas.append(out)
        return loss

    def validation_step(self, batch, batch_idx):
        """Phase 2: Validate LoRA model"""
        # Data preprocess
        img, emb, label = batch
        # Forward pass through student model
        student_features = self.model(img)  # student_features should have the same dimension as scLLM embedding
        student_logits = self.classifier(student_features)
        
        # Teacher logits are from pre-computed embeddings
        with torch.no_grad():
            teacher_logits = self.classifier(emb)
        
        # Compute loss
        loss = self.distillation_loss(student_logits, teacher_logits, label)
        if self.prototype_loss_weight > 0:
            prototype_loss, prototype_accuracy = self.prototype_loss(student_features, emb)
            loss = loss + self.prototype_loss_weight * prototype_loss
            self.log("val_prototype_loss", prototype_loss, on_step=False, on_epoch=True)
            self.log("val_prototype_accuracy", prototype_accuracy, on_step=False, on_epoch=True)
        out = self.val_post_process(student_logits, teacher_logits, loss)
        self.val_epoch_datas.append(out)
        return loss

    def on_save_checkpoint(self, checkpoint):
        """Save LoRA weights"""
        if self.lora_save_path is not None:
            print("save as a LoRA model and translate model")
            # Save LoRA weights
            self.model.encoder.save_pretrained(self.lora_save_path)
            # save translate model as normal nn.Module
            torch.save(self.model.translate_model, os.path.join(self.lora_save_path, "translate_model.pth"))
        else:
            print("save as a whole model")
            # save pl_model 
            super().on_save_checkpoint(checkpoint)

    def on_train_epoch_end(self):
        """Log metrics at the end of each training epoch"""
        if len(self.train_epoch_datas) > 0:
            self.log_train_metrics(self.train_epoch_datas, "train_loss")
        self.train_epoch_datas = []
        if self.prototype_loss_weight > 0:
            self._update_dynamic_prototypes()
            self.prototype_epoch_sums.zero_()
            self.prototype_epoch_counts.zero_()

    def on_validation_epoch_end(self):
        """Log metrics at the end of each validation epoch"""
        if len(self.val_epoch_datas) > 0:
            self.log_val_metrics(self.val_epoch_datas, "val_loss")
        self.val_epoch_datas = []
