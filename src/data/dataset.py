import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import pandas as pd
import os
from logs import logger
from torchvision import transforms
from src.utils.get_image_ps1 import get_ps1_image


class Space_dataset(Dataset):
    def __init__(
            self,
            path_csv: str,
            path_img: str,
            list_label: list[str],
            list_extrra: list[str] | None = None,
            transform: transforms.Compose | None = None
            ):
        """
        Создание датасет

        Args:
            path_csv: путь к csv данным
            path_img: путь к изображениям
            list_extrra: список столбцов для экстра признаков (например, ['z-factor'])
            list_label: список столбцов для меток
        Returns:
            Кортеж тензоров (изображение, метка)
        """
        # Проверка CSV
        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"CSV файл не найден: {path_csv}")
        self.path_csv = path_csv
        self.df = pd.read_csv(path_csv)  # Открываем csv файл
        if self.df.empty:
            raise ValueError("CSV файл пустой")
        if not os.path.exists(path_img):
            raise FileNotFoundError(f"Папка с изображение не найдена: {path_img}")
        self.path_img = path_img
        self.list_label = self.df[list_label] # Берем столбцы меток
        if list_extrra is not None:
            self.list_extrra = self.df[list_extrra] # Если есть экстра признаки берем их
        else:
            self.list_extrra = None
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item)  -> tuple[torch.Tensor,torch.Tensor] :
        ide = self.df['id'].iloc[item]

        ra, dec = self.df['ra'].iloc[item], self.df['dec'].iloc[item]  # берем ra и dec (координаты)
        if self.list_extrra:
            if not self.list_extrra.empty:
                extra = self.list_extrra.iloc[item]

        label = self.list_label.iloc[item].to_numpy()  # Берем метки и преобразовываем их в числа
        label_tensor = torch.tensor(label, dtype=torch.long)  # Теперь в тензор

        image_path = os.path.join(self.path_img, f"{ide}.jpg")

        if os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                image = ImageOps.exif_transpose(image)  # Корректная ориентация
                image.load()
            except (OSError, IOError) as e:
                logger.error(f"Не удалось открыть изображение {image_path}: {e}")
                raise RuntimeError(f"Broken image: {image_path}") from e
        else:
            matrix = get_ps1_image(ra,dec) # Если нет, то обращаемся к функции для получения этого изображения
            if matrix is None or matrix.size == 0:
                raise RuntimeError(f"Не удалось получить изображение для ra={ra}, dec={dec}")

            if matrix.dtype != np.uint8: # Нормализуем матрицу в [0, 255]

                if matrix.max() <= 1.0:
                    matrix = (matrix * 255).astype(np.uint8)
                else:
                    matrix = np.clip(matrix, 0, 255).astype(np.uint8)

            image = Image.fromarray(matrix) # Из матрицы делаем изображение
            image.save(image_path)
            print(f"Сохранили {image_path}")
            logger.debug(f"Сохранили {image_path}")

        if self.transform:
            image = self.transform(image)

        return image,label_tensor

    def __repr__(self):
        return f"Space_dataset(len={len(self)}, path_csv={self.path_csv})"