import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

import wandb
from evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_multilabel_classification_metrics,
)
from utils.config_utils import load_model, save_model, save_model_config


class Trainer:
    _instance = None

    def __new__(cls, logger: logging.Logger, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.logger = logger
            cls._instance.reset_global_variables()
        return cls._instance

    def reset_global_variables(self) -> None:
        self.global_best_val_loss = float("inf")
        self.global_best_val_f1 = 0
        self.global_best_model_path = None
        self.global_best_config = None
        self.logger.info(
            f"Global variables reset. Global best validation loss: {self.global_best_val_loss}, Global best validation F1: {self.global_best_val_f1}, Global best model path: {self.global_best_model_path}, Global best config: {self.global_best_config}"
        )

    def initialize_wandb(self, project_name: str, entity_name: str, config: Dict):
        wandb.init(project=project_name, entity=entity_name)
        for key, value in config.items():
            wandb.config[key] = value

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        device: str,
        use_wandb: bool = True,
    ) -> float:
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()

            inputs, attention_mask, labels = (
                batch["ids"].to(device),
                batch["mask"].to(device),
                batch["labels"].to(device),
            )
            outputs = model(inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if use_wandb:
                # Log the training loss to wandb
                wandb.log({"train/loss": loss.item()})

        return total_loss / len(dataloader)

    def validate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str,
        task_type: str,
        use_wandb: bool = True,
    ):
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs, attention_mask, labels = (
                    batch["ids"].to(device),
                    batch["mask"].to(device),
                    batch["labels"].to(device),
                )
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                if task_type == "multilabel":
                    predictions = torch.sigmoid(outputs.logits)
                    predictions[predictions >= 0.5] = 1
                    predictions[predictions < 0.5] = 0
                else:
                    predictions = torch.argmax(outputs.logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                if use_wandb:
                    # Log the validation loss to wandb
                    wandb.log({"val/loss": loss.item()})

        if task_type == "multilabel":
            (
                precision_macro,
                recall_macro,
                f1_macro,
                precision_weighted,
                recall_weighted,
                f1_weighted,
                precision_samples,
                recall_samples,
                f1_samples,
                hamming,
                accuracy,
            ) = compute_multilabel_classification_metrics(all_labels, all_predictions)
            return (
                total_loss / len(dataloader),
                precision_macro,
                recall_macro,
                f1_macro,
                precision_weighted,
                recall_weighted,
                f1_weighted,
                precision_samples,
                recall_samples,
                f1_samples,
                hamming,
                accuracy,
            )
        elif task_type == "multiclass":
            (
                precision_macro,
                recall_macro,
                f1_macro,
                precision_micro,
                recall_micro,
                f1_micro,
                precision_weighted,
                recall_weighted,
                f1_weighted,
                accuracy,
            ) = compute_multiclass_classification_metrics(all_labels, all_predictions)
            return (
                total_loss / len(dataloader),
                precision_macro,
                recall_macro,
                f1_macro,
                precision_micro,
                recall_micro,
                f1_micro,
                precision_weighted,
                recall_weighted,
                f1_weighted,
                accuracy,
            )
        else:
            precision, recall, f1, accuracy = compute_binary_classification_metrics(
                all_labels, all_predictions
            )
            return total_loss / len(dataloader), precision, recall, f1, accuracy

    def train_model(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        device: str,
        num_epochs: int,
        learning_rate: float,
        models_save_dir: str,
        early_stopping_patience: int,
        num_warmup_steps: int,
        config: Dict,
        task_type: str,
        use_wandb: bool = True,
    ) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
        """
        Trains the model using the provided data and hyperparameters.

        Args:
            model (torch.nn.Module): The model to be trained.
            train_dataloader (torch.utils.data.DataLoader): The data loader for training data.
            val_dataloader (torch.utils.data.DataLoader): The data loader for validation data.
            device (str): The device to be used for training (e.g., 'cuda', 'cpu').
            num_epochs (int): The number of training epochs.
            learning_rate (float): The learning rate for the optimizer.
            models_save_dir (str): The directory to save the trained models.
            early_stopping_patience (int): The number of epochs to wait for improvement before early stopping.
            num_warmup_steps (int): The number of warm-up steps for the learning rate scheduler.
            config (Dict): The configuration dictionary for the model.
            task_type (str): The type of task (e.g., 'multilabel', 'binary').
            use_wandb (bool, optional): Whether to use wandb for logging. Defaults to True.

        Returns:
            Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]: A tuple containing the trained model, optimizer, and additional information.
        """
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_epochs * len(train_dataloader)
        )

        best_val_loss = float("inf")
        best_val_f1 = 0
        epochs_without_improvement = 0
        best_model_path = None

        # Training loop
        for epoch in range(num_epochs):
            avg_train_loss = self.train_one_epoch(
                model, train_dataloader, optimizer, scheduler, device, use_wandb
            )

            if task_type == "multilabel":
                (
                    avg_val_loss,
                    val_precision_macro,
                    val_recall_macro,
                    val_f1_macro,
                    val_precision_weighted,
                    val_recall_weighted,
                    val_f1_weighted,
                    val_precision_samples,
                    val_recall_samples,
                    val_f1_samples,
                    val_hamming,
                    val_accuracy,
                ) = self.validate(model, val_dataloader, device, task_type, use_wandb)

                if use_wandb:
                    # Log validation metrics to wandb
                    wandb.log(
                        {
                            "train/avg_loss": avg_train_loss,
                            "val/avg_loss": avg_val_loss,
                            "val/precision_macro": val_precision_macro,
                            "val/recall_macro": val_recall_macro,
                            "val/f1_macro": val_f1_macro,
                            "val/precision_weighted": val_precision_weighted,
                            "val/recall_weighted": val_recall_weighted,
                            "val/f1_weighted": val_f1_weighted,
                            "val/precision_samples": val_precision_samples,
                            "val/recall_samples": val_recall_samples,
                            "val/f1_samples": val_f1_samples,
                            "val/hamming": val_hamming,
                            "val/accuracy": val_accuracy,
                        }
                    )

                self.logger.info(
                    f"Epoch {epoch}/{num_epochs - 1} - Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Val precision macro: {val_precision_macro}, Val recall macro: {val_recall_macro}, Val F1 macro: {val_f1_macro}, Val precision weighted: {val_precision_weighted}, Val recall weighted: {val_recall_weighted}, Val F1 weighted: {val_f1_weighted}, Val precision samples: {val_precision_samples}, Val recall samples: {val_recall_samples}, Val F1 samples: {val_f1_samples}, Val hamming: {val_hamming}, Val accuracy: {val_accuracy}"
                )
            elif task_type == "multiclass":
                (
                    avg_val_loss,
                    val_precision_macro,
                    val_recall_macro,
                    val_f1_macro,
                    val_precision_micro,
                    val_recall_micro,
                    val_f1_micro,
                    val_precision_weighted,
                    val_recall_weighted,
                    val_f1_weighted,
                    val_accuracy,
                ) = self.validate(model, val_dataloader, device, task_type, use_wandb)

                if use_wandb:
                    # Log validation metrics to wandb
                    wandb.log(
                        {
                            "train/avg_loss": avg_train_loss,
                            "val/avg_loss": avg_val_loss,
                            "val/precision_macro": val_precision_macro,
                            "val/recall_macro": val_recall_macro,
                            "val/f1_macro": val_f1_macro,
                            "val/precision_micro": val_precision_micro,
                            "val/recall_micro": val_recall_micro,
                            "val/f1_micro": val_f1_micro,
                            "val/precision_weighted": val_precision_weighted,
                            "val/recall_weighted": val_recall_weighted,
                            "val/f1_weighted": val_f1_weighted,
                            "val/accuracy": val_accuracy,
                        }
                    )

                self.logger.info(
                    f"Epoch {epoch}/{num_epochs - 1} - Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Val precision macro: {val_precision_macro}, Val recall macro: {val_recall_macro}, Val F1 macro: {val_f1_macro}, Val precision micro: {val_precision_micro}, Val recall micro: {val_recall_micro}, Val F1 micro: {val_f1_micro}, Val precision weighted: {val_precision_weighted}, Val recall weighted: {val_recall_weighted}, Val F1 weighted: {val_f1_weighted}, Val accuracy: {val_accuracy}"
                )
            else:
                avg_val_loss, val_precision, val_recall, val_f1, val_accuracy = self.validate(
                    model, val_dataloader, device, task_type, use_wandb
                )

                if use_wandb:
                    # Log validation metrics to wandb
                    wandb.log(
                        {
                            "train/avg_loss": avg_train_loss,
                            "val/avg_loss": avg_val_loss,
                            "val/precision": val_precision,
                            "val/recall": val_recall,
                            "val/f1": val_f1,
                            "val/accuracy": val_accuracy,
                        }
                    )

                self.logger.info(
                    f"Epoch {epoch}/{num_epochs - 1} - Train loss: {avg_train_loss}, Val loss: {avg_val_loss}, Val precision: {val_precision}, Val recall: {val_recall}, Val F1: {val_f1}, Val accuracy: {val_accuracy}"
                )

            # Checkpointing based on the validation loss
            if avg_val_loss < best_val_loss:
                self.logger.info(
                    f"Best validation loss changed from {best_val_loss} to {avg_val_loss}. Saving a checkpoint at epoch {epoch}/{num_epochs - 1}."
                )
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0  # reset the counter
                self.logger.info(f"Epochs without improvement counter: {epochs_without_improvement}")
                best_model_path = os.path.join(models_save_dir, "checkpoint.pt")
                self.logger.info(f"Saving the checkpoint model to {best_model_path}")
                save_model(
                    model, optimizer, {"epoch": epoch, "best_val_loss": best_val_loss}, best_model_path
                )
                save_model_config(models_save_dir, "checkpoint_config.json", config)

                # Check if this is the best model across all settings
                if avg_val_loss < self.global_best_val_loss:
                    self.logger.info(
                        f"Global best validation loss across all settings changed from {self.global_best_val_loss} to {avg_val_loss}. Saving the best model."
                    )
                    self.global_best_val_loss = avg_val_loss
                    self.global_best_config = config.copy()
                    self.global_best_model_path = os.path.join(models_save_dir, "best_model.pt")
                    self.logger.info(f"Saving the best model to {self.global_best_model_path}")
                    save_model(
                        model,
                        optimizer,
                        {"epoch": epoch, "best_val_loss": best_val_loss},
                        self.global_best_model_path,
                    )
                    save_model_config(models_save_dir, "best_model_config.json", config)
            else:
                epochs_without_improvement += 1
                self.logger.info(
                    f"Best validation loss did not change. Epochs without improvement counter: {epochs_without_improvement}"
                )

            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info("Training completed.")

        self.logger.info(f"Loading the best model from path {best_model_path}")
        return load_model(model, optimizer, best_model_path, device)
