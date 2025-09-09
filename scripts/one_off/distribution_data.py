import pandas as pd
import glob
import os

from pathlib import Path

def mix_classes_cyclically(folder,pattern,limit):
    folder = Path(folder)
    # Получаем файлы
    files = sorted(folder.glob(pattern))
    min_class = {}

    for filename in files[:limit]:
        df = pd.read_csv(filename)
        unique_class = df['cod_class'].value_counts()
        # Добавляем все пары класс → количество в словарь
        for cls, count in unique_class.items():
            min_class[cls] = min_class.get(cls, 0) + count
        output_file = r'D:\Code\Space_canvas\data\general_csv.csv'

        # Берём первый класс
        first_class = df['cod_class'].iloc[0]

        # Фильтруем DataFrame — оставляем только нужный класс
        df_filtered = df[df['cod_class'] == first_class]
        print(df_filtered)
        # # Проверяем, существует ли файл
        # file_exists = Path(output_file).is_file()
        #
        # # Записываем (добавляем в конец, если файл существует)
        # df_filtered.to_csv(
        #     output_file,
        #     mode='a' if file_exists else 'w',  # 'a' — добавить, 'w' — перезаписать (новый файл)
        #     index=False,
        #     header=not file_exists  # заголовок только если файл новый
        # )


pattern = "spall_chunk_*.csv"
folder = r"D:\Code\Space_canvas\data\spall_csv_chunks_encoded"
mix_classes_cyclically(folder,pattern,50)