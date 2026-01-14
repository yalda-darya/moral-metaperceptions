from typing import Dict, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class AbortionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        label_column: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.data = dataframe
        self.texts = dataframe[text_column].tolist()
        self.labels = torch.tensor(dataframe[label_column].tolist()).to(torch.long)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Dict[str, Union[torch.Tensor, int]]:
        text = str(self.texts[index])
        inputs = self.tokenizer(
            text=text,
            max_length=self.max_length,
            padding="longest",
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "ids": inputs["input_ids"].flatten(),
            "mask": inputs["attention_mask"].flatten(),
            "labels": self.labels[index],
        }


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
    labels = torch.tensor([item["labels"] for item in batch])

    return {"ids": ids, "mask": masks, "labels": labels}


def get_abortion_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int,
    max_length: int,
    model_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    train_dataset = AbortionDataset(train_df, "full_text", "label", tokenizer, max_length)
    val_dataset = AbortionDataset(val_df, "full_text", "label", tokenizer, max_length)
    test_dataset = AbortionDataset(test_df, "full_text", "label", tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Calculate the class weights
    train_df_class_freq = train_df["label"].value_counts().to_dict()
    train_df_num_classes = len(train_df["label"].unique())
    train_df_total_samples = len(train_df)

    # First tried way: Failed

    # train_df_class_weights = [
    #     train_df_total_samples / (train_df_num_classes * train_df_class_freq[i])
    #     for i in range(train_df_num_classes)
    # ]
    # train_df_class_weights = torch.tensor(train_df_class_weights, dtype=torch.float)

    # Second tried way
    beta = 0.999  # TODO: Make this a hyperparameter for wandb
    effective_num = {i: 1.0 - beta**freq for i, freq in train_df_class_freq.items()}
    weights = {i: (1 - beta) / (1 - beta**freq) for i, freq in effective_num.items()}
    train_df_class_weights = torch.tensor(
        [weights[i] for i in range(train_df_num_classes)], dtype=torch.float
    )

    return train_dataloader, val_dataloader, test_dataloader, train_df_class_weights
