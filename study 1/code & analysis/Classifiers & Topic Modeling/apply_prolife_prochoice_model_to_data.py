import os
from typing import Dict, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from utils.config_utils import get_device, load_model, load_model_config

ID_TO_LABEL = {0: "life", 1: "choice", 2: "neutral", 3: "throw_out"}


class BERTweetOutput:
    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss


class BERTweetClassifier(torch.nn.Module):
    def __init__(self, num_labels: int):
        super(BERTweetClassifier, self).__init__()

        self.num_labels = num_labels

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
            loss_fn = torch.nn.CrossEntropyLoss()
            # Calculate loss
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return BERTweetOutput(logits=logits, loss=loss)


class AbortionDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        text_column: str,
        tokenizer: AutoTokenizer,
        max_length: int,
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
            padding="longest",
            return_token_type_ids=True,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "ids": inputs["input_ids"].flatten(),
            "mask": inputs["attention_mask"].flatten(),
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
    best_model_dir = os.path.join(os.getcwd(), "best_prolife_prochoice_model")
    # best_model_dir = os.path.join(os.getcwd(), "best_binary_morality_model")

    best_model_config = load_model_config(best_model_dir, "best_model_config.json")

    max_length = best_model_config["max_length"]
    model_name = best_model_config["model_name"]
    model_num_labels = best_model_config["model_num_labels"]
    batch_size = best_model_config["batch_size"]
    early_stopping_patience = best_model_config["early_stopping_patience"]
    learning_rate = best_model_config["learning_rate"]
    num_epochs = best_model_config["num_epochs"]
    num_warmup_steps = best_model_config["num_warmup_steps"]

    device = get_device()

    if model_name == "vinai/bertweet-base":
        model = BERTweetClassifier(model_num_labels).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=model_num_labels
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model, optimizer, checkpoint = load_model(
        model, optimizer, os.path.join(best_model_dir, "best_model.pt"), device
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")

    abortion_dataset_complete_df = pd.read_csv(
        os.path.join(os.getcwd(), "data", "raw", "abortion_dataset_complete.csv"),
        engine="python",
        dtype={
            "full_text": "str",
            "conversation_id_str": "str",
            "created_at": "object",
            "id_str": "str",
            "annotation": "str",
            "source": "str",
        },
    )
    abortion_dataset_complete_df["full_text"] = abortion_dataset_complete_df["full_text"].str.strip()

    # dtype of full_text, id_str, created_at, and annotation are string
    abortion_dataset_complete_df["full_text"] = abortion_dataset_complete_df["full_text"].astype(str)
    abortion_dataset_complete_df["id_str"] = abortion_dataset_complete_df["id_str"].astype(str)
    abortion_dataset_complete_df["annotation"] = abortion_dataset_complete_df["annotation"].astype(str)

    print(f"Number of rows in the dataset: {abortion_dataset_complete_df.shape[0]}")

    # Get the rows with annotation equal to empty string
    abortion_dataset_complete_without_annotation_df = abortion_dataset_complete_df[
        abortion_dataset_complete_df["annotation"] == "nan"
    ]

    print(f"Number of rows without annotation: {abortion_dataset_complete_without_annotation_df.shape[0]}")

    assert abortion_dataset_complete_without_annotation_df["annotation"].isin(["nan"]).all()

    # Change the annotation column name to "prolife_prochoice"
    abortion_dataset_complete_without_annotation_df.rename(
        columns={"annotation": "prolife_prochoice"}, inplace=True
    )

    # Get the rows with annotation not equal to empty string
    abortion_dataset_complete_with_annotation_df = abortion_dataset_complete_df[
        abortion_dataset_complete_df["annotation"] != "nan"
    ]

    print(f"Number of rows with annotation: {abortion_dataset_complete_with_annotation_df.shape[0]}")

    # assert every value in the annotation column is in in the ID_TO_LABEL dictionary keys
    assert abortion_dataset_complete_with_annotation_df["annotation"].isin(ID_TO_LABEL.values()).all()

    # Change the annotation column name to "prolife_prochoice"
    abortion_dataset_complete_with_annotation_df.rename(
        columns={"annotation": "prolife_prochoice"}, inplace=True
    )

    test_dataloader = get_abortion_dataloaders(
        abortion_dataset_complete_without_annotation_df,
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
            predictions = torch.argmax(outputs.logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())

    # Now that we have the predictions, we can update the df
    abortion_dataset_complete_without_annotation_df.loc[:, "prolife_prochoice"] = all_predictions
    abortion_dataset_complete_without_annotation_df.loc[:, "prolife_prochoice"] = (
        abortion_dataset_complete_without_annotation_df["prolife_prochoice"].map(ID_TO_LABEL)
    )

    # Now we can combine the two dfs
    df = pd.concat(
        [abortion_dataset_complete_without_annotation_df, abortion_dataset_complete_with_annotation_df],
        ignore_index=True,
    )

    df = df.reset_index(drop=True)

    df.to_csv(
        os.path.join(
            os.getcwd(), "data", "raw", "abortion_dataset_complete_with_prolife_prochoice_labels.csv"
        ),
        index=False,
    )
