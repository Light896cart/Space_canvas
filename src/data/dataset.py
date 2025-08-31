import torch
from torch.utils.data import Dataset
import pandas as pd


class Space_dataset(Dataset):
    def __init__(self,path_csv:str,path_img:str,list_label=None,list_x=None):
        """
        :param path_csv: путь к CSV с метаданными
        :param path_img: путь к изображениям (может не использоваться сейчас)
        :param list_x: список столбцов для признаков (например, ['cod_class', 'cod_subclass'])
        :param list_label: список столбцов для меток (если нужны отдельно)
        """
        df = pd.read_csv(path_csv)
        self.list_label = df[list_x].values
        self.list_x = df[list_x]

    def __len__(self):
        return len(self.df)
    def __getitem__(self, item):
        x = self.list_x.iloc[item]
        x_tensor = torch.tensor(x, dtype=torch.long)  # или float, если нужно
        return x_tensor


path_csv = r'D:\Code\Space_canvas\data\spall_csv_chunks_encoded\spall_chunk_0001.csv'
path_img = r'D:\Code\Space_canvas\data\image_data\img_csv\408595600150005914600060103.jpg'
reg = Space_dataset(path_csv=path_csv,path_img=path_img,list_x=['cod_class','cod_subclass'])
for c in reg:
    print(c)
    break