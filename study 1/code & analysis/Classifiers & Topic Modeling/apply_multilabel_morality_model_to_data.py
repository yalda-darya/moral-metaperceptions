import os
from typing import Dict, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils.config_utils import get_device, load_model, load_model_config

ID_TO_LABEL = {0: "care", 1: "purity", 2: "loyalty", 3: "authority", 4: "fairness"}


class AbortionDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, text_column: str, tokenizer: AutoTokenizer, max_length: int
    ) -> None:
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe[text_column].tolist()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        text = str(self.texts[index])
        inputs = self.tokenizer(
            text=text,
            max_length=self.max_length,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        return {"ids": inputs["input_ids"].flatten(), "mask": inputs["attention_mask"].flatten()}


def collate_fn(batch):
    # Find the longest sequence in the batch
    max_length = max(len(item["ids"]) for item in batch)

    # Pad each item in the batch to max_length
    for item in batch:
        padding_length = max_length - len(item["ids"])
        item["ids"] = torch.nn.functional.pad(item["ids"], (0, padding_length), value=0)
        item["mask"] = torch.nn.functional.pad(item["mask"], (0, padding_length), value=0)

    # Stack the items in the batch
    ids = torch.stack([item["ids"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])

    return {"ids": ids, "mask": masks}


def get_abortion_dataloaders(
    dataframe: pd.DataFrame,
    batch_size: int,
    max_length: int,
    tokenizer: AutoTokenizer,
) -> DataLoader:

    test_dataset = AbortionDataset(dataframe, "full_text", tokenizer, max_length)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return test_dataloader


if __name__ == "__main__":
    best_model_dir = os.path.join(os.getcwd(), "best_multilabel_morality_model")

    best_model_config = load_model_config(best_model_dir, "best_model_config.json")

    max_length = best_model_config["max_length"]
    model_name = best_model_config["model_name"]
    model_num_labels = best_model_config["model_num_labels"]

    # batch_size = best_model_config["batch_size"]
    batch_size = 32

    early_stopping_patience = best_model_config["early_stopping_patience"]
    learning_rate = best_model_config["learning_rate"]
    num_epochs = best_model_config["num_epochs"]
    num_warmup_steps = best_model_config["num_warmup_steps"]

    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=model_num_labels, problem_type="multi_label_classification"
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer, checkpoint = load_model(
        model, optimizer, os.path.join(best_model_dir, "best_model.pt"), device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")

    abortion_dataset_complete_df = pd.read_csv(
        os.path.join(
            os.getcwd(),
            "data",
            "raw",
            "abortion_dataset_complete_with_prolife_prochoice_binary_morality_labels.csv",
        ),
        engine="python",
        dtype={
            "full_text": "str",
            "conversation_id_str": "str",
            "created_at": "object",
            "id_str": "str",
            "user": "object",
            "prolife_prochoice": "str",
            "binary_morality": "str",
        },
    )
    abortion_dataset_complete_df["full_text"] = abortion_dataset_complete_df["full_text"].str.strip()

    # dtype of full_text, id_str, created_at, prolife_prochoice, and binary_morality are string
    abortion_dataset_complete_df["full_text"] = abortion_dataset_complete_df["full_text"].astype(str)
    abortion_dataset_complete_df["id_str"] = abortion_dataset_complete_df["id_str"].astype(str)
    abortion_dataset_complete_df["prolife_prochoice"] = abortion_dataset_complete_df[
        "prolife_prochoice"
    ].astype(str)
    abortion_dataset_complete_df["binary_morality"] = abortion_dataset_complete_df["binary_morality"].astype(
        str
    )

    print(f"Number of rows in the dataset: {abortion_dataset_complete_df.shape[0]}")

    # assert the prolife_prochoice column has only ['choice', 'life', 'neutral', 'throw_out'] values
    assert (
        abortion_dataset_complete_df["prolife_prochoice"]
        .isin(["choice", "life", "neutral", "throw_out"])
        .all()
    )

    # assert the binary_morality column has only ['moral', 'non-moral'] values
    assert abortion_dataset_complete_df["binary_morality"].isin(["moral", "non-moral"]).all()

    # For each moral value, we will have a separate column initially filled with 0s
    for label in ID_TO_LABEL.values():
        abortion_dataset_complete_df[label] = 0

    # Make a new dataframe that has binary_morality == "moral" named abortion_dataset_moral_df
    abortion_dataset_moral_df = abortion_dataset_complete_df[
        abortion_dataset_complete_df["binary_morality"] == "moral"
    ].reset_index(drop=True)

    print(f"Number of rows in the moral dataset: {abortion_dataset_moral_df.shape[0]}")

    # Make a new dataframe that has binary_morality == "non-moral" named abortion_dataset_non_moral_df
    abortion_dataset_non_moral_df = abortion_dataset_complete_df[
        abortion_dataset_complete_df["binary_morality"] == "non-moral"
    ].reset_index(drop=True)

    print(f"Number of rows in the non-moral dataset: {abortion_dataset_non_moral_df.shape[0]}")

    # Apply the model to the moral dataset
    test_dataloader = get_abortion_dataloaders(
        abortion_dataset_moral_df,
        batch_size=batch_size,
        max_length=max_length,
        tokenizer=tokenizer,
    )

    model.eval()
    all_predictions = []

    # Evaluate on the test set
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            inputs, attention_mask = (
                batch["ids"].to(device),
                batch["mask"].to(device),
            )

            outputs = model(inputs, attention_mask)
            predictions = torch.sigmoid(outputs.logits)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
            all_predictions.extend(predictions.cpu().numpy())

    # Now that we have the predictions, we can update the moral value columns in the moral dataframe
    for idx, label in ID_TO_LABEL.items():
        abortion_dataset_moral_df.loc[:, label] = [int(pred[idx]) for pred in all_predictions]

    # Concatenate the moral and non-moral dataframes
    abortion_dataset_complete_df = pd.concat(
        [abortion_dataset_moral_df, abortion_dataset_non_moral_df], ignore_index=True
    )

    # assert that the dtype of the moral value columns is int (0 or 1)
    for label in ID_TO_LABEL.values():
        assert abortion_dataset_complete_df[label].isin([0, 1]).all()

    abortion_dataset_complete_df = abortion_dataset_complete_df.reset_index(drop=True)

    abortion_dataset_complete_df.to_csv(
        os.path.join(
            os.getcwd(),
            "data",
            "raw",
            "abortion_dataset_complete_with_prolife_prochoice_binary_morality_multimodal_morality_labels.csv",
        ),
        index=False,
    )
