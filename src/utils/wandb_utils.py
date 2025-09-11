# src/utils/wandb_utils.py
import os
import atexit
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Optional

def init_wandb(cfg: DictConfig, project_name: str = "space-ml", job_type: str = "train") -> None:
    """
    Инициализирует W&B run в offline режиме с сохранением логов локально.
    """
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # Явно указываем, где будут храниться offline-логи
    os.makedirs("./wandb", exist_ok=True)  # Убедимся, что папка есть

    wandb.init(
        project=project_name,
        config=config_dict,
        job_type=job_type,
        mode="disabled",
    )