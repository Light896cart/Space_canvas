import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os

from src.utils.get_image_ps1 import get_ps1_image


class Space_dataset(Dataset):
    def __init__(self,path_csv:str,path_img:str,list_label=None,list_x=None):
        """
        :param path_csv: путь к CSV с метаданными
        :param path_img: путь к изображениям (может не использоваться сейчас)
        :param list_x: список столбцов для признаков (например, ['cod_class', 'cod_subclass'])
        :param list_label: список столбцов для меток (если нужны отдельно)
        """
        self.path_img = path_img
        self.df = pd.read_csv(path_csv)
        self.list_label = self.df[list_label]
        self.list_x = self.df[list_x]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        ide = self.df['id'].iloc[item]
        extra = self.list_x.iloc[item]
        ra, dec = self.list_x['ra'].iloc[item], self.list_x['dec'].iloc[item]
        label = self.list_label.iloc[item].to_numpy()
        label_tensor = torch.tensor(label, dtype=torch.long)  # или float, если нужно
        image_path = os.path.join(self.path_img, f"{ide}.jpg")
        print(image_path)
        # Проверяем, есть ли файл
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            matrix = get_ps1_image(ra,dec)

            if matrix.dtype != np.uint8:
                if matrix.max() <= 1.0:
                    matrix = (matrix * 255).astype(np.uint8)
                else:
                    matrix = np.clip(matrix, 0, 255).astype(np.uint8)
            # Преобразуем матрицу в изображение
            image = Image.fromarray(matrix)
            image.save(image_path)
        print('Итерация')
        # plt.imshow(image)
        # plt.show()
        return image,label_tensor

path_csv = r'D:\Code\Space_canvas\data\spall_csv_chunks_encoded\spall_chunk_0001.csv'
path_img = r'D:\Code\Space_canvas\data\image_data\img_csv'
reg = Space_dataset(path_csv=path_csv,path_img=path_img,list_label=['cod_class','cod_subclass'],list_x=['ra','dec'])
for c in reg:
    print(c)
