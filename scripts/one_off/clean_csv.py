"""
⚠️ ONE-OFF SCRIPT
Цель: однократная очистка колонки 'subclass' в CSV-датасете.
Не предназначен для переиспользования.
Запущен: 2025-04-05, данные обработаны.
"""
import os
import pandas as pd
import re


def clean_csv(path_dir_csv, inplace=True, output_dir=None):
    """
    Очищает колонку 'subclass' в CSV-файлах: удаляет всё в скобках, например (12345).
    Сохраняет файлы с обновлёнными значениями.

    Args:
        path_dir_csv (str): Путь к папке с CSV-файлами.
        inplace (bool): Если True — перезаписывает файлы. Если False — сохраняет в output_dir.
        output_dir (str): Путь к папке для сохранения очищенных файлов (нужен если inplace=False).
    """

    # Собираем все CSV-файлы
    csv_files = [f for f in os.listdir(path_dir_csv) if f.lower().endswith('.csv')]

    if not csv_files:
        print("❌ Нет CSV-файлов для обработки.")
        return

    print(f"🔍 Найдено {len(csv_files)} CSV-файлов.")

    # Функция очистки subclass
    def clean_subclass(sub):
        if pd.isna(sub) or sub == "" or sub == "unknown":
            return "unknown"

        sub = str(sub).strip()

        # 1. Удаляем всё, что в скобках с пробелом: ' (12345)'
        sub = sub.split(' (')[0]

        # 2. Удаляем многоточие в любом месте (можно только в конце: \.+$)
        sub = sub.replace('...', '').replace('..', '').replace('.', '').strip()

        # Дополнительно: удаляем другие шумовые символы (опционально)
        # sub = re.sub(r'[?$+:^*]', '', sub).strip()  # если нужно

        # Если после очистки пусто — считаем неизвестным
        if not sub or sub == "":
            return "unknown"

        return sub

    # Определяем, куда сохранять
    save_dir = path_dir_csv if inplace else output_dir
    if not inplace and not output_dir:
        raise ValueError("Если inplace=False, нужно указать output_dir.")

    if not inplace:
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Очищенные файлы будут сохранены в: {output_dir}")

    # Обработка каждого файла
    for file in csv_files:
        filepath = os.path.join(path_dir_csv, file)
        try:
            df = pd.read_csv(filepath)

            # Проверяем, есть ли колонка subclass
            if 'subclass' not in df.columns:
                print(f"⚠️ Пропущен файл {file}: нет колонки 'subclass'")
                continue

            # Сохраняем оригинальный тип данных
            original_dtype = df['subclass'].dtype

            # Очищаем subclass
            df['subclass'] = df['subclass'].astype(str).apply(clean_subclass)

            # Восстанавливаем тип (опционально; в данном случае лучше оставить str)
            # → оставляем как str, потому что категории

            # Сохраняем
            output_path = os.path.join(save_dir, file)
            df.to_csv(output_path, index=False)
            print(f"✅ Очищен и сохранён: {file}")

        except Exception as e:
            print(f"❌ Ошибка при обработке {file}: {e}")


# ———————————————————————————————————
# 🚀 Пример вызова
# ———————————————————————————————————

# Вариант 1: перезаписать файлы на том же месте
# clean_csv(path_dir_csv=r"D:\Code\Space_canvas\data\spall_csv_chunks_lazy", inplace=True)

# Вариант 2: сохранить в новую папку (безопаснее!)
clean_csv(
    path_dir_csv=r"D:\Code\Space_canvas\data\spall_csv_chunks_lazy",
    inplace=False,
    output_dir=r"D:\Code\Space_canvas\data\spall_csv_chunks_cleaned"
)