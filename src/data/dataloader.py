import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.data.dataset import Space_dataset


def create_train_val_dataloader(
        path_csv: str,
        path_img: str,
        list_label: list[str],
        train_ratio: int | None = 0.9,
        list_extra: list[str] | None = None,
        transform: transforms.Compose | None = None,
):
    """
    Разделяем на тренировочные и валидационные данные, и создаем DataLoader

    Args:
        path_csv: путь до файла csv
        path_img: путь до папки с изображениями
        list_label: список с названиями столбцов с метками
        train_ratio: процент тренировочных данных от всего датасета
        list_extra: список с названиями столбцов с дополнительными признаками
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
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Разделяем
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset = DataLoader(train_dataset,batch_size=32,shuffle=True)
    val_dataset = DataLoader(val_dataset,batch_size=32,shuffle=False)
    return train_dataset, val_dataset