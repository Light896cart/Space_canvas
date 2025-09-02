import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.data.dataset import Space_dataset


def create_train_val_dataloader(
        path_csv: str,
        path_img: str,
        list_label: list[str],
        list_extra: list[str] | None = None,
        transform: transforms.Compose | None = None,
):
    """
    Разделяем на тренировочные и валидационные данные, и создаем DataLoader

    Args:
        path_csv: путь до файла csv
        path_img: путь до папки с изображениями
        list_label: список с названиями столбцов с метками
        list_extra: список с названиеями столбцов с дополнительными признакми
        transform: трансформер

    Return:

    """

    # Пример данных
    dataset = Space_dataset(
        path_csv=path_csv,
        path_img=path_img,
        list_label=list_label,
        list_extrra=list_extra,
        transform=transform
    )
    # Задаём пропорции
    train_ratio = 0.8
    val_ratio = 0.2
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Разделяем
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset = DataLoader(train_dataset,batch_size=32,shuffle=True)
    return train_dataset