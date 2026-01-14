import inspect
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Union, Tuple

import torch
import yaml


def get_dataset_full_path(task_name: str, random_state: int, dataset_name: str) -> str:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(
        project_dir, "data", "processed", task_name, f"random_state_{random_state}", dataset_name
    )
    return (
        os.path.join(data_dir, "train.csv"),
        os.path.join(data_dir, "validation.csv"),
        os.path.join(data_dir, "test.csv"),
    )


def get_model_save_full_path(
    task_name: str,
    random_state: int,
    dataset_name: str,
    model_name: str,
    variation: [None | str] = None,
) -> str:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(
        project_dir,
        "models",
        task_name,
        f"random_state_{random_state}",
        dataset_name,
        model_name,
        variation,
    )
    return models_dir


def get_device(logger: logging.Logger = None, device_index: int = 0) -> str:
    # Check that MPS is available
    if not torch.backends.mps.is_available():
        device = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"

        # Clear the cache
        torch.cuda.empty_cache()
    else:
        device = "mps"

    if logger is not None:
        logger.info(f"Using device: {device}")
    else:
        print(f"Using device: {device}")
    return device


def setup_logging(logs_dir: str = "logs") -> logging.Logger:
    # Ensure the logs directory exists
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Get the module that called this function name
    stack = inspect.stack()
    module_name = stack[1][1].split("/")[-1].split(".")[0]

    # Generate the log filename based on the module name and the current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{module_name}_{current_time}.log"
    log_filepath = os.path.join(logs_dir, log_filename)

    # Set up the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Set the root logger level

    # Create a file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)

    return logger


def load_config(config_path: str) -> Union[Dict[Any, Any], None]:
    with open(config_path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def save_model_config(models_save_dir: str, file_name: str, config: Dict) -> None:
    with open(os.path.join(models_save_dir, file_name), "w") as f:
        json.dump(config, f, indent=4)


def load_model_config(models_save_dir: str, file_name: str) -> Dict:
    with open(os.path.join(models_save_dir, file_name), "r") as f:
        return json.load(f)


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    other_info: Dict[str, Union[int, float, str]],
    path: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **other_info,
        },
        path,
    )


def load_model(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: str, device: str
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint
