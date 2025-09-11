import os
from pathlib import Path

import wandb
from omegaconf import DictConfig
import hydra
from torchvision import transforms

from src.data.dataloader import create_train_val_dataloader
from src.data.dataset import Space_dataset
from logs import logger
from src.model.learn_model import train_model
from src.utils.seeding import set_seed
from src.utils.wandb_utils import init_wandb
# Определяем корень и путь к configs
project_root = Path(__file__).parent.parent
config_path = str(project_root / "configs")

os.chdir(project_root)

transform = transforms.Compose([
    transforms.ToTensor(),               # обязательно!
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # опционально
])

@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    """
    Основная точка входа в проект.

    Загружает конфигурацию, устанавливает seed, создаёт dataloader
    и выполняет предварительную проверку данных.
    Args:
        cfg (DictConfig): Конфигурация из Hydra. Ожидается:
            - cfg.data.csv_path: путь к CSV
            - cfg.data.path_img: путь к изображениям
            - cfg.data.list_label: список столбцов меток
            - cfg.data.list_extra: дополнительные признаки (опционально)
            - cfg.seed: seed для воспроизводимости
            - Продолжение следует...

    Returns:
        None: функция ничего не возвращает, только выполняет побочные эффекты
               (создание dataloader, логирование, обучение и т.д.)

    Raises:
        FileNotFoundError: если CSV или папка с изображениями не найдены
        RuntimeError: если ошибка при создании dataloader
        ValueError: если некорректные параметры в конфиге
    """
    folder = cfg.data.folder
    path_img = cfg.data.path_img
    path_val_dataset = cfg.data.path_val_dataset
    seed = cfg.seed
    list_label = cfg.data.list_label
    list_extra = cfg.data.list_extra

    # 👇 Инициализируем W&B для отслеживание метрик
    init_wandb(cfg)

    set_seed(seed)
    logger.info('Создаем датасет')
    train_model(
        folder=folder,
        list_label=list_label,
        path_img=path_img,
        list_extra=list_extra,
        transform=transform,
        path_val_dataset=path_val_dataset
    )

    # ⚠ Завершаем run
    wandb.finish()
    # # # dataset = Space_dataset(path_csv=path_csv,path_img=img_path,list_label=['cod_class','cod_subclass'],list_x=['ra','dec'])
    # # # print("CSV Path:", dataset)

if __name__ == "__main__":
    main()