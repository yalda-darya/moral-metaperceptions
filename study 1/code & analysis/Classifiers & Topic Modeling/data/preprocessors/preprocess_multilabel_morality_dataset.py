import argparse
import logging
import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from utils.config_utils import load_config, setup_logging


def load_mftc_data(mftc_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(mftc_filepath, usecols=["tweet_text"] + [f"annotation_{i}" for i in range(1, 9)])
    return df


def load_mfrc_data(mfrc_filepath: str) -> pd.DataFrame:
    df = pd.read_csv(mfrc_filepath, usecols=["text", "annotator", "annotation", "confidence"])
    return df


def clean_mftc_data(mftc_df: pd.DataFrame) -> pd.DataFrame:
    mftc_df["tweet_text"] = mftc_df["tweet_text"].str.strip()

    for col in mftc_df.columns:
        if col.startswith("annotation"):
            mftc_df[col] = mftc_df[col].str.lower()

    return mftc_df


def clean_mfrc_data(mfrc_df: pd.DataFrame) -> pd.DataFrame:
    mfrc_df["annotation"] = mfrc_df["annotation"].str.lower()
    mfrc_df["annotation"] = mfrc_df["annotation"].str.split(",")

    return mfrc_df


def transform_mftc_data(mftc_df: pd.DataFrame) -> pd.DataFrame:
    mftc_df["moral_count"] = 0
    mftc_df["non_moral_count"] = 0

    for col in mftc_df.columns:
        if col.startswith("annotation"):
            mftc_df[col] = mftc_df[col].fillna("")
            mftc_df[col] = mftc_df[col].str.replace("harm", "care")
            mftc_df[col] = mftc_df[col].str.replace("cheating", "fairness")
            mftc_df[col] = mftc_df[col].str.replace("betrayal", "loyalty")
            mftc_df[col] = mftc_df[col].str.replace("subversion", "authority")
            mftc_df[col] = mftc_df[col].str.replace("degradation", "purity")
            mftc_df[col] = mftc_df[col].str.replace("nh", "non-moral")
            mftc_df[col] = mftc_df[col].str.replace("nm", "non-moral")
            mftc_df[col] = mftc_df[col].str.split(",")
            mftc_df[col] = mftc_df[col].apply(lambda x: [] if x == [""] else x)

            mftc_df[col] = mftc_df[col].apply(lambda x: list(set(x)))

            mftc_df["moral_count"] += mftc_df[col].apply(
                lambda x: 1 if len(set(x)) >= 1 and "non-moral" not in x else 0
            )
            mftc_df["non_moral_count"] += mftc_df[col].apply(
                lambda x: 1 if len(set(x)) == 1 and "non-moral" in x else 0
            )

    mftc_df["binary_label"] = mftc_df.apply(
        lambda row: "moral"
        if row["moral_count"] > row["non_moral_count"]
        else "non-moral"
        if row["non_moral_count"] > row["moral_count"]
        else "tie",
        axis=1,
    )
    mftc_df = mftc_df[mftc_df["binary_label"] != "tie"].reset_index(drop=True)

    # Keep only the rows that are 'moral' in the binary_label column
    mftc_df = mftc_df[mftc_df["binary_label"] == "moral"].reset_index(drop=True)
    # Create a "labels" column, which will keep all of the labels from the annotation columns except 'non-moral' and NaN
    mftc_df["labels"] = mftc_df.apply(
        lambda row: set(
            [
                label
                for label in row[
                    [
                        "annotation_1",
                        "annotation_2",
                        "annotation_3",
                        "annotation_4",
                        "annotation_5",
                        "annotation_6",
                        "annotation_7",
                        "annotation_8",
                    ]
                ].explode()
                if label != "non-moral" and pd.isna(label) == False
            ]
        ),
        axis=1,
    )
    return mftc_df


def transform_mfrc_data(mfrc_df: pd.DataFrame) -> pd.DataFrame:
    mfrc_df["annotation"] = mfrc_df["annotation"].apply(
        lambda x: [label.replace("equality", "fairness") if label == "equality" else label for label in x]
    )
    mfrc_df["annotation"] = mfrc_df["annotation"].apply(
        lambda x: [
            label.replace("proportionality", "fairness") if label == "proportionality" else label
            for label in x
        ]
    )

    mfrc_df["annotation_00"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator00" else [], axis=1
    )
    mfrc_df["annotation_01"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator01" else [], axis=1
    )
    mfrc_df["annotation_02"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator02" else [], axis=1
    )
    mfrc_df["annotation_03"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator03" else [], axis=1
    )
    mfrc_df["annotation_04"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator04" else [], axis=1
    )
    mfrc_df["annotation_05"] = mfrc_df.apply(
        lambda row: row["annotation"] if row["annotator"] == "annotator05" else [], axis=1
    )

    mfrc_df = mfrc_df.drop_duplicates(subset=["text", "annotator"], keep="first").reset_index(drop=True)

    mfrc_df = (
        mfrc_df.groupby("text")
        .agg(
            {
                "annotation_00": "sum",
                "annotation_01": "sum",
                "annotation_02": "sum",
                "annotation_03": "sum",
                "annotation_04": "sum",
                "annotation_05": "sum",
            }
        )
        .reset_index()
    )

    mfrc_df["moral_count"] = 0
    mfrc_df["non_moral_count"] = 0

    for col in mfrc_df.columns:
        if "annotation_" in col:
            mfrc_df["moral_count"] += mfrc_df[col].apply(
                lambda x: 1 if len(set(x)) >= 1 and "non-moral" not in x else 0
            )
            mfrc_df["non_moral_count"] += mfrc_df[col].apply(
                lambda x: 1 if len(set(x)) == 1 and "non-moral" in x else 0
            )

    mfrc_df["binary_label"] = mfrc_df.apply(
        lambda row: "moral"
        if row["moral_count"] > row["non_moral_count"]
        else "non-moral"
        if row["non_moral_count"] > row["moral_count"]
        else "tie",
        axis=1,
    )
    mfrc_df = mfrc_df[mfrc_df["binary_label"] != "tie"].reset_index(drop=True)

    # Keep only the rows that are 'moral' in the binary_label column
    mfrc_df = mfrc_df[mfrc_df["binary_label"] == "moral"].reset_index(drop=True)

    # Create a "labels" column, which will keep all of the labels from the annotation columns except 'non-moral' and 'thin morality'
    mfrc_df["labels"] = mfrc_df.apply(
        lambda row: set(
            [
                label
                for label in row[
                    [
                        "annotation_00",
                        "annotation_01",
                        "annotation_02",
                        "annotation_03",
                        "annotation_04",
                        "annotation_05",
                    ]
                ].explode()
                if label != "non-moral" and label != "thin morality" and pd.isna(label) == False
            ]
        ),
        axis=1,
    )

    # Drop the rows where the labels column is empty
    mfrc_df = mfrc_df[mfrc_df["labels"].apply(lambda x: len(x) > 0)].reset_index(drop=True)

    return mfrc_df


def merge_mftc_mfrc(
    mftc_df: pd.DataFrame, mfrc_df: pd.DataFrame, label_map: Dict[str, int], random_state: int
) -> pd.DataFrame:
    # From mftc only keep the tweet_text and binary_label columns, rename the tweet_text column to text
    mftc_df = mftc_df[["tweet_text", "labels"]]
    mftc_df = mftc_df.rename(columns={"tweet_text": "text"})

    # From mfrc only keep the text and binary_label columns
    mfrc_df = mfrc_df[["text", "labels"]]

    # Concatenate mftc_df and mfrc_df
    df = pd.concat([mftc_df, mfrc_df], ignore_index=True)

    # Create a dictionary mapping of all unique labels to a number
    label_to_number = {label: i for i, label in enumerate(df["labels"].explode().unique())}

    df["numeric_labels"] = df["labels"].apply(lambda labels: [label_to_number[label] for label in labels])

    mlb = MultiLabelBinarizer(classes=sorted(label_to_number.values()))
    df["labels_encoded"] = list(mlb.fit_transform(df["numeric_labels"]))

    return df


def split_data(df: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Splitting data
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=(1 / 9), random_state=random_state)
    return train_df, val_df, test_df


def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, path: str) -> None:
    # Saving processed data
    train_df.to_csv(os.path.join(path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(path, "validation.csv"), index=False)
    test_df.to_csv(os.path.join(path, "test.csv"), index=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the configuration file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    config = load_config(args.config)

    setup_logging(config["logs_dir"])

    for random_state in config["random_states"]:
        PROCESSED_DATA_PATH = os.path.join(
            config["output_path"],
            f"random_state_{random_state}",
            config["dataset_name"],
        )
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

        logging.info("Starting data preprocessing...")
        logging.info(f"MFTC path: {config['mftc_input_path']}")
        logging.info(f"MFRC path: {config['mfrc_input_path']}")
        logging.info(f"Output path: {PROCESSED_DATA_PATH}")
        logging.info(f"Random state: {random_state}")

        mftc_df = load_mftc_data(config["mftc_input_path"])
        mfrc_df = load_mfrc_data(config["mfrc_input_path"])

        mftc_df = clean_mftc_data(mftc_df)
        mfrc_df = clean_mfrc_data(mfrc_df)

        mftc_df = transform_mftc_data(mftc_df)
        mfrc_df = transform_mfrc_data(mfrc_df)

        df = merge_mftc_mfrc(mftc_df, mfrc_df, config["label_map"], random_state)

        logging.info(f"{config['label_map']=}")

        train_df, val_df, test_df = split_data(df, random_state)
        save_data(train_df, val_df, test_df, PROCESSED_DATA_PATH)
        logging.info(f"Data preprocessing completed for random state: {random_state}")
