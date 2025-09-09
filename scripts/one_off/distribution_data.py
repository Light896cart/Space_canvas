import pandas as pd
import glob
import os

from pathlib import Path

def mix_classes_cyclically(folder,pattern,limit):
    folder = Path(folder)
    # Получаем файлы
    files = sorted(folder.glob(pattern))
    min_class = {}
    first_class = 0

    for filename in files[:limit]:
        df = pd.read_csv(filename)
        unique_class = df['cod_class'].value_counts()
        # Добавляем все пары класс → количество в словарь
        for cls, count in unique_class.items():
            min_class[cls] = min_class.get(cls, 0) + count

    output_file = r'D:\Code\Space_canvas\data\general_csv.csv'

    changed = True
    min_value = min(min_class.values())
    global_min_value = min_value
    while changed:
        changed = False  # Сбрасываем перед началом прохода
        print("Новый цикл со значением",first_class)
        for filename in files[:limit]:
            df = pd.read_csv(filename)
            # Фильтруем DataFrame — оставляем только нужный класс
            df_filtered = df[df['cod_class'] == first_class]
            min_value -= len(df_filtered)
            # Проверяем, существует ли файл
            file_exists = Path(output_file).is_file()

            # Записываем (добавляем в конец, если файл существует)
            df_filtered[:min_value].to_csv(
                output_file,
                mode='a' if file_exists else 'w',  # 'a' — добавить, 'w' — перезаписать (новый файл)
                index=False,
                header=not file_exists  # заголовок только если файл новый
            )
            if min_value <= 0:
                min_value = global_min_value
                first_class += 1
                changed = True
                break
            if len(df_filtered) == 0:
                break


pattern = "spall_chunk_*.csv"
folder = r"D:\Code\Space_canvas\data\spall_csv_chunks_encoded"
mix_classes_cyclically(folder,pattern,50)