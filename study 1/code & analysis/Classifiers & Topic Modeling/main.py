import argparse
import logging
import os
from typing import Callable

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification

import wandb
from data.loaders.abortion_dataset_loader import get_abortion_dataloaders
from data.loaders.moral_multilabel_dataset_loader import get_moral_multilabel_dataloaders
from evaluation.metrics import (
    compute_binary_classification_metrics,
    compute_multiclass_classification_metrics,
    compute_multilabel_classification_metrics,
)
from train.trainer import Trainer
from utils.config_utils import (
    get_dataset_full_path,
    get_device,
    get_model_save_full_path,
    load_config,
    setup_logging,
)


class BERTweetOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class BERTweetClassifier(torch.nn.Module):
    def __init__(self, num_labels: int, train_class_weights: torch.Tensor = None):
        super(BERTweetClassifier, self).__init__()

        self.num_labels = num_labels
        self.train_class_weights = train_class_weights

        # Load the BERTweet model
        self.bertweet = AutoModel.from_pretrained("vinai/bertweet-base")

        # Define a classification layer
        self.classifier = torch.nn.Linear(self.bertweet.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Pass inputs through BERTweet model
        outputs = self.bertweet(input_ids=input_ids, attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs.last_hidden_state[:, 0, :]

        # Pass the last hidden state of the token `[CLS]` to the classifier layer
        logits = self.classifier(last_hidden_state_cls)

        # Calculate loss only if labels are provided
        loss = None
        if labels is not None:
            # Convert labels to long data type
            labels = labels.long()

            # Define the loss function
            loss_fn = torch.nn.CrossEntropyLoss(weight=self.train_class_weights)
            # Calculate loss
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return BERTweetOutput(logits=logits, loss=loss)


def make_sweep_runner(
    train_path: str,
    val_path: str,
    test_path: str,
    max_length: int,
    model_name: str,
    model_num_labels: int,
    device: str,
    logger: logging.Logger,
    models_save_dir: str,
    task_type: str,
) -> Callable[[], None]:
    def run_sweep():
        wandb.init()
        # Name the run using the parameters
        wandb.run.name = (
            f"lr_{wandb.config.learning_rate}_bs_{wandb.config.batch_size}_epochs_{wandb.config.num_epochs}"
        )

        # Call main with the provided and fetched parameters
        main(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=wandb.config.batch_size,
            max_length=max_length,
            model_name=model_name,
            model_num_labels=model_num_labels,
            device=device,
            epochs=wandb.config.num_epochs,
            learning_rate=wandb.config.learning_rate,
            logger=logger,
            models_save_dir=models_save_dir,
            early_stopping_patience=wandb.config.early_stopping_patience,
            num_warmup_steps=wandb.config.num_warmup_steps,
            task_type=task_type,
        )

    return run_sweep


def main(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    max_length: int,
    model_name: str,
    model_num_labels: int,
    device: str,
    epochs: int,
    learning_rate: float,
    logger: logging.Logger,
    models_save_dir: str,
    early_stopping_patience: int,
    num_warmup_steps: int,
    task_type: str,
    use_wandb: bool = True,
) -> None:
    # Create a comprehensive config dictionary
    config = {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "max_length": max_length,
        "model_name": model_name,
        "model_num_labels": model_num_labels,
        "device": device,
        "model_save_dir": models_save_dir,
    }

    if use_wandb:
        for key, value in config.items():
            wandb.config[key] = value

        # Add wandb configs to the dictionary
        config.update(wandb.config._items)

    if task_type == "multilabel":
        # Load data
        train_loader, val_loader, test_loader = get_moral_multilabel_dataloaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            max_length=max_length,
            model_name=model_name,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=model_num_labels, problem_type="multi_label_classification"
        ).to(device)
    else:
        # Load data
        train_loader, val_loader, test_loader, train_class_weights = get_abortion_dataloaders(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            batch_size=batch_size,
            max_length=max_length,
            model_name=model_name,
        )

        if model_name == "vinai/bertweet-base":
            if train_class_weights is not None:
                train_class_weights = train_class_weights.to(device)
            model = BERTweetClassifier(model_num_labels, train_class_weights=train_class_weights).to(device)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=model_num_labels
            ).to(device)

    # Train the model
    model, optimizer, checkpoint = trainer.train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        num_epochs=epochs,
        learning_rate=learning_rate,
        models_save_dir=models_save_dir,
        early_stopping_patience=early_stopping_patience,
        num_warmup_steps=num_warmup_steps,
        config=config,
        task_type=task_type,
        use_wandb=use_wandb,
    )

    # Evaluate the model (you can expand this with more detailed evaluation if needed)
    all_true_labels, all_predictions = [], []

    # Evaluate on the test set
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, attention_mask, labels = (
                batch["ids"].to(device),
                batch["mask"].to(device),
                batch["labels"].to(device),
            )

            outputs = model(inputs, attention_mask=attention_mask)

            if task_type == "multilabel":
                predictions = torch.sigmoid(outputs.logits)
                predictions[predictions >= 0.5] = 1
                predictions[predictions < 0.5] = 0
            else:
                predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

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
        ) = compute_multilabel_classification_metrics(all_true_labels, all_predictions)
        logger.info(f"Test Precision (macro): {precision_macro}")
        logger.info(f"Test Recall (macro): {recall_macro}")
        logger.info(f"Test F1 (macro): {f1_macro}")
        logger.info(f"Test Precision (weighted): {precision_weighted}")
        logger.info(f"Test Recall (weighted): {recall_weighted}")
        logger.info(f"Test F1 (weighted): {f1_weighted}")
        logger.info(f"Test Precision (samples): {precision_samples}")
        logger.info(f"Test Recall (samples): {recall_samples}")
        logger.info(f"Test F1 (samples): {f1_samples}")
        logger.info(f"Test Hamming Loss: {hamming}")
        logger.info(f"Test Accuracy: {accuracy}")

        if use_wandb:
            # Log the test metrics to wandb
            wandb.log(
                {
                    "test/precision_macro": precision_macro,
                    "test/recall_macro": recall_macro,
                    "test/f1_macro": f1_macro,
                    "test/precision_weighted": precision_weighted,
                    "test/recall_weighted": recall_weighted,
                    "test/f1_weighted": f1_weighted,
                    "test/precision_samples": precision_samples,
                    "test/recall_samples": recall_samples,
                    "test/f1_samples": f1_samples,
                    "test/hamming": hamming,
                    "test/accuracy": accuracy,
                }
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
        ) = compute_multiclass_classification_metrics(all_true_labels, all_predictions)
        logger.info(f"Test Precision (macro): {precision_macro}")
        logger.info(f"Test Recall (macro): {recall_macro}")
        logger.info(f"Test F1 (macro): {f1_macro}")
        logger.info(f"Test Precision (micro): {precision_micro}")
        logger.info(f"Test Recall (micro): {recall_micro}")
        logger.info(f"Test F1 (micro): {f1_micro}")
        logger.info(f"Test Precision (weighted): {precision_weighted}")
        logger.info(f"Test Recall (weighted): {recall_weighted}")
        logger.info(f"Test F1 (weighted): {f1_weighted}")
        logger.info(f"Test Accuracy: {accuracy}")

        if use_wandb:
            # Log the test metrics to wandb
            wandb.log(
                {
                    "test/precision_macro": precision_macro,
                    "test/recall_macro": recall_macro,
                    "test/f1_macro": f1_macro,
                    "test/precision_micro": precision_micro,
                    "test/recall_micro": recall_micro,
                    "test/f1_micro": f1_micro,
                    "test/precision_weighted": precision_weighted,
                    "test/recall_weighted": recall_weighted,
                    "test/f1_weighted": f1_weighted,
                    "test/accuracy": accuracy,
                }
            )
    else:
        # Then, you'd call the evaluation function:
        precision, recall, f1, accuracy = compute_binary_classification_metrics(
            all_true_labels, all_predictions
        )
        logger.info(f"Test Precision: {precision}")
        logger.info(f"Test Recall: {recall}")
        logger.info(f"Test F1: {f1}")
        logger.info(f"Test Accuracy: {accuracy}")

        if use_wandb:
            # Log the test metrics to wandb
            wandb.log(
                {
                    "test/precision": precision,
                    "test/recall": recall,
                    "test/f1": f1,
                    "test/accuracy": accuracy,
                }
            )

    if use_wandb:
        # Finish the run
        wandb.finish()


def get_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the moral values classifier")
    parser.add_argument(
        "--training_config",
        required=True,
        help="Path to the training configuration file.",
    )
    parser.add_argument("--model_config", required=True, help="Path to the model configuration file.")
    parser.add_argument("--sweep_config", required=True, help="Path to the sweep configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    training_config = load_config(args.training_config)
    model_config = load_config(args.model_config)

    logger = setup_logging(training_config["logs_dir"])

    trainer = Trainer(logger)

    for random_state in training_config["random_states"]:
        for dataset in training_config["dataset_names"]:
            device = get_device(logger)
            train_path, val_path, test_path = get_dataset_full_path(
                training_config["task_name"], random_state, dataset
            )

            # Make sure the model_name does not contain any forward slashes and replace them with underscores
            models_save_dir = get_model_save_full_path(
                training_config["wandb"]["project"],
                random_state,
                dataset,
                model_config["model_name"].replace("/", "_"),
                training_config["variation"],
            )
            os.makedirs(models_save_dir, exist_ok=True)

            if training_config["wandb"]["use_sweep"] == True:
                sweep_config = load_config(args.sweep_config)

                # Add the training config to the sweep config
                if training_config["variation"] is not None:
                    sweep_config["name"] = f"{sweep_config['name']}-{training_config['variation']}"

                sweep_id = wandb.sweep(
                    sweep=sweep_config,
                    project=training_config["wandb"]["project"],
                    entity=training_config["wandb"]["entity"],
                )

                # Create the sweep runner function with all the required parameters
                sweep_runner = make_sweep_runner(
                    train_path=train_path,
                    val_path=val_path,
                    test_path=test_path,
                    max_length=model_config["max_input_length"],
                    model_name=model_config["model_name"],
                    model_num_labels=training_config["model_num_labels"],
                    device=device,
                    logger=logger,
                    models_save_dir=models_save_dir,
                    task_type=training_config["task_type"],
                )
                wandb.agent(
                    sweep_id,
                    function=sweep_runner,
                    count=training_config["wandb"]["count"],
                )

                # Stop the agent after the sweep is finished
                os.system(
                    f"wandb sweep --stop {training_config['wandb']['entity']}/{training_config['wandb']['project']}/{sweep_id}"
                )

                trainer.reset_global_variables()

            else:
                main(
                    train_path=train_path,
                    val_path=val_path,
                    test_path=test_path,
                    batch_size=8,
                    max_length=model_config["max_input_length"],
                    model_name=model_config["model_name"],
                    model_num_labels=training_config["model_num_labels"],
                    device=device,
                    epochs=4,
                    learning_rate=1e-6,
                    logger=logger,
                    models_save_dir=models_save_dir,
                    early_stopping_patience=2,
                    num_warmup_steps=30,
                    task_type=training_config["task_type"],
                    use_wandb=False,
                )
