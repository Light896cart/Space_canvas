import pandas as pd
import glob
import os

from pathlib import Path

def mix_classes_cyclically(
        folder: str,
        pattern: str,
        limit: int,
        name_class: str,
        output_file: str,
        min_value: int | None = None
):
    """
    Создаем csv файл с идеальным распределение данных, для тренировочных данных или валидационных (преимущественно)

    Args:
        folder: Путь до папки с csv файлами (датасет) (данные уже должны быть закодированы)
        pattern: Паттерн csv файла. Например: spall_chunk_*.csv (вместо звездочки будут подставляться цифры)
        limit: Какое кол во файлов мы возьмем
        name_class: Относительно какого столбца мы будем делать идеальное распределение
        output_file: По какому пути мы будем сохранять новый csv файл. Например: 'D:\Code\Space_canvas\data\general_csv.csv'
        min_value: Какое кол во уникальных классов мы хотим увидеть. Например: всего 3 класса, и по 100 вариантов одного
        уникального класса
    Return:
        None
    """
    folder = Path(folder)
    # Получаем файлы
    files = sorted(folder.glob(pattern))
    min_class = {}
    first_class = 0

    for filename in files[:limit]:
        df = pd.read_csv(filename)
        unique_class = df[name_class].value_counts()
        # Добавляем все пары класс → количество в словарь
        for cls, count in unique_class.items():
            min_class[cls] = min_class.get(cls, 0) + count

    if not min_value:
        min_value = min(min_class.values())
    global_min_value = min_value
    for cls in min_class:
        print("Новый цикл со значением",cls)
        for filename in files[:limit]:
            df = pd.read_csv(filename)
            # Фильтруем DataFrame — оставляем только нужный класс
            df_filtered = df[df[name_class] == cls]
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
                break
            if len(df_filtered) == 0:
                break
    df = pd.read_csv(output_file)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(output_file, index=False)


pattern = "spall_chunk_*.csv"
folder = r"D:\Code\Space_canvas\data\val_csv_dataset"
output_file = r'D:\Code\Space_canvas\data\val_dataset.csv'
mix_classes_cyclically(
    folder=folder,
    pattern=pattern,
    limit=50,
    name_class='cod_class',
    output_file=output_file,
    min_value=400
)