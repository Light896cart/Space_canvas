import pandas as pd
from pathlib import Path

def mix_classes_cyclically(
        folder: str,
        pattern: str,
        limit: int,
        name_class: str,
        output_file: str,
        min_value: int | None = None
):
    folder = Path(folder)
    files = sorted(folder.glob(pattern))[:limit]

    # Собираем все данные в один DataFrame
    all_data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Группируем по классам
    grouped = all_data.groupby(name_class)

    # Определяем минимум, если не задан
    if min_value is None:
        min_value = grouped.size().min()

    # Берём по min_value строк из каждого класса
    balanced_data = pd.concat([
        group.sample(n=min(len(group), min_value), random_state=42)
        for name, group in grouped
    ], ignore_index=True)

    # Перемешиваем
    balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Сохраняем
    balanced_data.to_csv(output_file, index=False)

pattern = "chunk_*.csv"
folder = r"D:\Code\Space_canvas\data\spall_csv_chunks_encoded"
output_file = r'D:\Code\Space_canvas\data\train_perfect_imbalance\chunk_0001.csv'

mix_classes_cyclically(
    folder=folder,
    pattern=pattern,
    limit=100,
    name_class='cod_class',
    output_file=output_file,
    min_value=1000
)