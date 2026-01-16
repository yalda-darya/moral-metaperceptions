import ast
from typing import Dict, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class MoralMultilabelDataset(Dataset):
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
        self.labels = torch.tensor(
            dataframe[label_column].str.replace(" ", ",").apply(ast.literal_eval), dtype=torch.float
        )
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
    labels = torch.stack([item["labels"] for item in batch])

    return {"ids": ids, "mask": masks, "labels": labels}


def get_moral_multilabel_dataloaders(
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

    train_dataset = MoralMultilabelDataset(train_df, "text", "labels_encoded", tokenizer, max_length)
    val_dataset = MoralMultilabelDataset(val_df, "text", "labels_encoded", tokenizer, max_length)
    test_dataset = MoralMultilabelDataset(test_df, "text", "labels_encoded", tokenizer, max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader
