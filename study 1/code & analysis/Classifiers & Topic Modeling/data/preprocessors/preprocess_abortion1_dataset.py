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
    df["full_text"] = df["full_text"].str.strip()
    df["annotation"] = df["annotation"].str.strip()
    df["annotation"] = df["annotation"].str.lower()
    df = df[~df["annotation"].isna()]

    return df


def transform_data(df: pd.DataFrame, label_map: Dict[str, int]) -> pd.DataFrame:
    df.loc[:, "annotation"] = df.loc[:, "annotation"].replace(
        to_replace=["neutral", "throw out"], value="throw_out"
    )
    df.loc[:, "label"] = df.loc[:, "annotation"].map(label_map)
    return df


def split_data(df: pd.DataFrame, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Splitting data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=random_state)
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
        logging.info(f"Inputs path: {config['input_path']}")
        logging.info(f"Output path: {PROCESSED_DATA_PATH}")
        logging.info(f"Random state: {random_state}")

        df = load_data(config["input_path"])
        df = clean_data(df)
        df = transform_data(df, config["label_map"])

        logging.info(f"Balanced dataset: {df['label'].value_counts()}")

        train_df, val_df, test_df = split_data(df, random_state)
        save_data(train_df, val_df, test_df, PROCESSED_DATA_PATH)
        logging.info(f"Data preprocessing completed for random state: {random_state}")
