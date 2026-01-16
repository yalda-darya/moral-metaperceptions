import argparse
import logging
import os
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config_utils import load_config, setup_logging


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    if "text" in df.columns:
        df = df.rename(columns={"text": "full_text"})
    df["full_text"] = df["full_text"].str.strip()
    df["annotation"] = df["annotation"].str.strip()
    df["annotation"] = df["annotation"].str.lower()
    df = df[~df["annotation"].isna()]

    return df


def transform_data(df: pd.DataFrame, label_map: Dict[str, int]) -> pd.DataFrame:
    # Change every "throw out" label to "throw_out" under the "annotation" column
    df.loc[df["annotation"] == "throw out", "annotation"] = "throw_out"
    # Remove all rows with throw_out label
    # df = df.loc[df["annotation"] != "throw_out"].copy()
    df = df.loc[df["annotation"].isin(label_map.keys())].copy()
    assert df["annotation"].isin(label_map.keys()).all()
    df.loc[:, "label"] = df["annotation"].map(label_map)
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
        logging.info(f"Abortion1 path: {config['abortion1_input_path']}")
        logging.info(f"Abortion2 path: {config['abortion2_input_path']}")
        logging.info(f"Output path: {PROCESSED_DATA_PATH}")
        logging.info(f"Random state: {random_state}")

        abortion1_df = load_data(config["abortion1_input_path"])
        abortion2_df = load_data(config["abortion2_input_path"])

        abortion1_df = clean_data(abortion1_df)
        abortion2_df = clean_data(abortion2_df)

        abortion1_df = transform_data(abortion1_df, config["label_map"])
        abortion2_df = transform_data(abortion2_df, config["label_map"])

        df = pd.concat([abortion1_df, abortion2_df], ignore_index=True)

        logging.info(f"{df['label'].value_counts()=}")

        train_df, val_df, test_df = split_data(df, random_state)
        save_data(train_df, val_df, test_df, PROCESSED_DATA_PATH)
        logging.info(f"Data preprocessing completed for random state: {random_state}")
