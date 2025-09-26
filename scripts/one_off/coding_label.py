"""
⚠️ ONE-OFF SCRIPT
Цель: однократная очистка колонки 'subclass' в CSV-датасете.
Не предназначен для переиспользования.
Запущен: 2025-04-05, данные обработаны.
"""
import os
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import pickle

def clean_subclass(sub):
    if pd.isna(sub) or sub == "unknown":
        return "unknown"
    # Делим по ' (' и берём первую часть
    cleaned = str(sub).split(' (')[0]
    return cleaned.strip()

def coding_label_create(path_csv, path_output, n_jobs=-1, verbose=1):
    """
    Обрабатывает все CSV-файлы в папке: кодирует class и subclass, сохраняет с новыми колонками.

    Args:
        path_csv (str): Путь к папке с CSV-файлами.
        path_output (str): Путь к папке для сохранения закодированных файлов.
        n_jobs (int): Количество параллельных процессов (-1 = все ядра).
        verbose (int): Уровень логирования.
    """

    # Создаём выходную папку
    os.makedirs(path_output, exist_ok=True)

    # Собираем все CSV-файлы
    csv_files = [f for f in os.listdir(path_csv) if f.lower().endswith('.csv')]
    if not csv_files:
        print("❌ Нет CSV-файлов для обработки.")
        return

    print(f"🔍 Найдено {len(csv_files)} CSV-файлов.")

    # Читаем все файлы (лениво — только имена и первые части, чтобы собрать уникальные метки)
    print("🔄 Собираем уникальные значения class и subclass для согласованного кодирования...")

    all_classes = []
    all_subclasses = []

    for file in csv_files:
        df_chunk = pd.read_csv(os.path.join(path_csv, file), usecols=['class', 'subclass'],
                               nrows=100000)  # ограничим для скорости
        all_classes.extend(df_chunk['class'].dropna().unique())
        all_subclasses.extend(df_chunk['subclass'].dropna().unique())

    # Уникализируем
    all_classes = list(set(all_classes))
    all_subclasses = list(set(all_subclasses))
    if "unknown" not in all_classes:
        all_classes.append("unknown")
    if "unknown" not in all_subclasses:
        all_subclasses.append("unknown")
    # Создаём LabelEncoder'ы
    le_class = LabelEncoder()
    le_subclass = LabelEncoder()

    le_class.fit(all_classes)
    le_subclass.fit(all_subclasses)
    print('all_classes', le_class)
    print('all_subclasses', le_subclass)
    # Сохраняем энкодеры (на будущее, если понадобится декодировать или обрабатывать новые данные)
    with open(os.path.join(path_output, 'label_encoder_class.pkl'), 'wb') as f:
        pickle.dump(le_class, f)
    with open(os.path.join(path_output, 'label_encoder_subclass.pkl'), 'wb') as f:
        pickle.dump(le_subclass, f)

    print(f"✅ Энкодеры обучены: {len(all_classes)} уникальных class, {len(all_subclasses)} subclass.")

    # Функция обработки одного файла
    def process_file(filename):
        try:
            filepath = os.path.join(path_csv, filename)
            df = pd.read_csv(filepath)

            # Проверяем нужные колонки
            if 'class' not in df.columns or 'subclass' not in df.columns:
                print(f"⚠️ Пропущен файл {filename}: нет колонок class/subclass")
                return

            # Кодируем (с обработкой NaN)
            df['cod_class'] = le_class.transform(df['class'].fillna("unknown"))
            df['cod_subclass'] = le_subclass.transform(df['subclass'].fillna("unknown"))

            # Сохраняем
            output_path = os.path.join(path_output, filename)
            df.to_csv(output_path, index=False)

            if verbose >= 2:
                print(f"✅ Обработан: {filename}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")

    # Параллельная обработка всех файлов
    print(f"🚀 Запускаем обработку {len(csv_files)} файлов...")
    Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(process_file)(f) for f in csv_files)

    print(f"✅ Готово! Закодированные файлы сохранены в: {path_output}")


# ———————————————————————————————————
# 🚀 Вызов функции
# ———————————————————————————————————

path_csv = r"D:\Code\Space_canvas\data\raw_csv"
path_output = r"D:\Code\Space_canvas\data\encoded_csv"
n_jobs = -1
verbose = 1

coding_label_create(path_csv, path_output, n_jobs=-1, verbose=1)