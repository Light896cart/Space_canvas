import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.dataset import Space_dataset


def create_train_val_dataloader(img_path,path_img):
    # Пример данных
    dataset = Space_dataset(path_csv=img_path,path_img=path_img,list_label=['cod_class','cod_subclass'],list_x=['ra','dec'])
    # Задаём пропорции
    train_ratio = 0.8
    val_ratio = 0.2
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    # Разделяем
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(train_dataset)