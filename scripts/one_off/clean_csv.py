"""
⚠️ ONE-OFF SCRIPT
Цель: однократная очистка колонки 'subclass' в CSV-файлах + удаление строк с пустыми id/ra/dec.
Если после обработки файл пуст (0 строк), он удаляется.
Не предназначен для переиспользования.
Запущен: 2025-04-05, данные обработаны.
"""

import os
import pandas as pd


def clean_csv(path_dir_csv, inplace=True, output_dir=None):
    """
    Очищает CSV-файлы в указанной директории:
      1. Удаляет строки, где 'id', 'ra' или 'dec' — пустые (NaN, None, '', пробелы).
      2. Очищает колонку 'subclass': удаляет содержимое в скобках и многоточия.
      3. Если после обработки в файле 0 строк — файл НЕ сохраняется (удаляется из вывода).
      4. Сохраняет только непустые файлы.

    Args:
        path_dir_csv (str): Путь к папке с CSV-файлами.
        inplace (bool): Если True — перезаписывает исходные файлы. Если False — сохраняет в output_dir.
        output_dir (str): Путь к папке для сохранения очищенных файлов (обязателен при inplace=False).
    """

    # Собираем все CSV-файлы
    csv_files = [f for f in os.listdir(path_dir_csv) if f.lower().endswith('.csv')]

    if not csv_files:
        print("❌ Нет CSV-файлов для обработки.")
        return

    print(f"🔍 Найдено {len(csv_files)} CSV-файлов.")

    # Функция очистки значения subclass
    def clean_subclass(sub):
        if pd.isna(sub) or sub == "" or sub == "unknown":
            return "unknown"

        sub = str(sub).strip()

        # Удаляем всё, что после ' (' — например: "QSO (12345)" → "QSO"
        sub = sub.split(' (')[0]

        # Удаляем точки и многоточия
        sub = sub.replace('...', '').replace('..', '').replace('.', '').strip()

        # Если после очистки пусто — считаем неизвестным
        return "unknown" if not sub else sub

    # Определяем директорию для сохранения
    save_dir = path_dir_csv if inplace else output_dir
    if not inplace and not output_dir:
        raise ValueError("Если inplace=False, необходимо указать параметр output_dir.")

    if not inplace:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Очищенные файлы будут сохранены в: {output_dir}")

    # Обработка каждого файла
    for file in csv_files:
        filepath = os.path.join(path_dir_csv, file)
        try:
            df = pd.read_csv(filepath)

            required_cols = ['id', 'ra', 'dec', 'subclass']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ Пропущен файл {file}: отсутствуют колонки {missing_cols}")
                continue

            print(f"📊 Обработка {file}: исходное количество строк — {len(df)}")

            # Функция проверки на пустое значение
            def is_empty(val):
                if pd.isna(val):
                    return True
                if isinstance(val, str) and val.strip() == "":
                    return True
                return False

            # Создаём маску: True — строку оставляем
            mask = ~(
                df['id'].apply(is_empty) |
                df['ra'].apply(is_empty) |
                df['dec'].apply(is_empty)
            )

            n_dropped = len(df) - mask.sum()
            df = df[mask].copy().reset_index(drop=True)

            print(f"🧹 Удалено строк с пустыми id/ra/dec: {n_dropped} → осталось {len(df)} строк")

            # Очищаем subclass, если есть строки
            if len(df) > 0:
                df['subclass'] = df['subclass'].astype(str).apply(clean_subclass)
            else:
                print(f"🗑️ Все строки удалены в файле {file}. Файл будет пропущен (не сохранён).")
                continue  # ⬅️ Не сохраняем, если строк нет

            # Сохраняем ТОЛЬКО если df не пустой
            output_path = os.path.join(save_dir, file)
            df.to_csv(output_path, index=False)
            print(f"✅ Обработан и сохранён: {file}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {file}: {e}")


# ———————————————————————————————————
# 🚀 Пример вызова
# ———————————————————————————————————

if __name__ == "__main__":
    # Вариант 1: перезаписать файлы на том же месте
    # clean_csv(path_dir_csv=r"D:\Code\Space_canvas\data\spall_csv_chunks_lazy", inplace=True)

    # Вариант 2: безопасно — сохранить в новую папку
    clean_csv(
        path_dir_csv=r"D:\Code\Space_canvas\data\spall_csv_chunks_encoded",
        inplace=False,
        output_dir=r"D:\Code\Space_canvas\data\spall_csv_chunks_cleaneder"
    )