import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.ndimage import median_filter
from torch.utils.data import Dataset
import pandas as pd
import os
from logs import logger
from torchvision import transforms
from src.utils.get_image_ps1 import get_ps1_multiband


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

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        ide = self.df['id'].iloc[item]
        ra, dec = self.df['ra'].iloc[item], self.df['dec'].iloc[item]
        label = self.list_label.iloc[item].to_numpy()
        label_tensor = torch.tensor(label, dtype=torch.long)

        npy_path = os.path.join(self.path_img, f"{ide}.npy")

        if os.path.exists(npy_path):
            matrix = np.load(npy_path)
        else:
            matrix = get_ps1_multiband(ra, dec)
            nan_count = np.isnan(matrix).sum()
            if nan_count > 0:
                matrix = fill_nan_with_median(matrix, kernel_size=3)
                # Проверяем, что теперь всё хорошо
                # Проверяем, что теперь всё хорошо
                if matrix is None or np.isnan(matrix).any():
                    print('К сожалению пусто ')
                    matrix = np.zeros((75, 75, 5), dtype=np.float32)
            if matrix is None:
                raise RuntimeError("Не удалось загрузить изображение")
            np.save(npy_path, matrix)
        # Превращаем в тензор: (H, W, 5) -> (5, H, W)
        image = torch.tensor(matrix, dtype=torch.float32).permute(2, 0, 1)
        return image, label_tensor

    def __repr__(self):
        return f"Space_dataset(len={len(self)}, path_csv={self.path_csv})"

def fill_nan_with_median(matrix, kernel_size=5):
    """
    Заменяет NaN на медиану окрестности (3x3), только для одного фильтра за раз.
    Безопасно для астрономических изображений.
    """
    matrix_filled = matrix.copy()
    for i in range(matrix.shape[2]):  # по каждому фильтру
        mask = np.isnan(matrix[:, :, i])
        if not mask.any():
            continue
        # Применяем медианную фильтрацию
        filled = median_filter(matrix[:, :, i], size=kernel_size, mode='nearest')
        matrix_filled[:, :, i][mask] = filled[mask]
    return matrix_filled