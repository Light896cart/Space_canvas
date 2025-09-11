import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from src.data.dataset import Space_dataset

# Глобальный кеш для валидационных датасетов (ключ — путь к CSV)
_VAL_DATASET_CACHE = {}

def create_train_val_dataloader(
        path_csv: str,
        path_img: str,
        list_label: list[str],
        train_ratio: int | None = 0.9,
        list_extra: list[str] | None = None,
        transform: transforms.Compose | None = None,
        path_val_dataset: str | None = None
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
        path_val_dataset: Путь до файла с валидационными данными
    Return:

    """

    # Пример данных
    dataset_train = Space_dataset(
        path_csv=path_csv,
        path_img=path_img,
        list_label=list_label,
        list_extrra=list_extra,
        transform=transform
    )
    if path_val_dataset:
        # Проверяем, есть ли уже такой датасет в кеше
        cache_key = (path_val_dataset, path_img, tuple(list_label), tuple(list_extra or []), id(transform))
        if cache_key in _VAL_DATASET_CACHE:
            dataset_val = _VAL_DATASET_CACHE[cache_key]
        else:
            dataset_val = Space_dataset(
                path_csv=path_val_dataset,
                path_img=path_img,
                list_label=list_label,
                list_extrra=list_extra,
                transform=transform
            )
            _VAL_DATASET_CACHE[cache_key] = dataset_val
    else:
        # Задаём пропорции
        train_size = int(train_ratio * len(dataset_train))
        val_size = len(dataset_train) - train_size

        # Разделяем
        dataset_train, dataset_val = random_split(dataset_train, [train_size, val_size])

    train_dataset = DataLoader(dataset_train,batch_size=32,shuffle=True)
    val_dataset = DataLoader(dataset_val,batch_size=32,shuffle=False)
    return train_dataset, val_dataset
