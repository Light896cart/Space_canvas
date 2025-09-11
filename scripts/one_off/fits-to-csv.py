# -*- coding: utf-8 -*-
"""
Ленивое чтение spAll-v6_1_3.fits — только:
id, ra, dec, class, subclass, z, ebv
Без PSFMAG и других фотометрических массивов.
"""

import os
import numpy as np
import pandas as pd
from astropy.io import fits


def lazy_create_csv_chunks(fits_filename, chunk_size=1000, output_dir='spall_csv_chunks'):
    """
    Читает spAll FITS и создаёт CSV с минимальным набором:
    - ID (ранее specobjid) и координаты
    - Класс и подкласс
    - Красное смещение и поглощение
    """

    # Создаём папку
    os.makedirs(output_dir, exist_ok=True)
    print(f"Открываем {fits_filename}...")

    with fits.open(fits_filename, memmap=True) as hdul:
        hdu = hdul[1]  # BinTableHDU

        # Определяем количество строк
        try:
            total_rows = len(hdu.data)
        except TypeError:
            total_rows = hdu.data.shape[0]

        print(f"Общее количество строк: {total_rows}")

        # Список всех колонок
        colnames = [col.name for col in hdu.columns]

        # 🔹 Только нужные признаки
        col_mapping = {
            'specobjid': ['SPECOBJID'],   # будем читать как specobjid, но сохранять как id
            'ra': ['PLUG_RA', 'RACAT'],
            'dec': ['PLUG_DEC', 'DECCAT'],
            'class': ['CLASS'],
            'subclass': ['SUBCLASS'],
            'z': ['Z'],
            'ebv': ['EBV']
        }

        # Поиск индексов
        col_idx = {}
        required = ['specobjid', 'ra', 'dec', 'class', 'z']
        for key, variants in col_mapping.items():
            col_idx[key] = None
            for name in variants:
                if name in colnames:
                    col_idx[key] = colnames.index(name)
                    print(f"✅ {key:<10} -> '{name}'")
                    break
            if col_idx[key] is None and key in required:
                print(f"❌ Критическая ошибка: не найдено поле '{key}'")
                return

        # --- Чтение и запись ---
        file_counter = 1
        buffer = []

        for i in range(total_rows):
            try:
                record = hdu.data[i]
            except Exception:
                continue  # пропускаем битые строки

            # Собираем строку с переименованным id
            row = {}

            # specobjid → сохраняем как 'id'
            idx = col_idx['specobjid']
            if idx is not None:
                val = record[idx]
                if isinstance(val, float) and not np.isfinite(val):
                    val = None
                row['id'] = val  # ⬅️ Вот он — ключевое изменение
            else:
                row['id'] = None

            # Остальные поля
            for key in ['ra', 'dec', 'class', 'subclass', 'z', 'ebv']:
                idx = col_idx.get(key)
                if idx is not None:
                    val = record[idx]
                    if isinstance(val, float) and not np.isfinite(val):
                        val = None
                    row[key] = val
                else:
                    row[key] = None

            buffer.append(row)

            # Сохраняем чанк
            if len(buffer) >= chunk_size:
                filename = os.path.join(output_dir, f"chunk_{file_counter:04d}.csv")
                pd.DataFrame(buffer).to_csv(filename, index=False)
                print(f"💾 Сохранено: {filename}")
                buffer = []
                file_counter += 1

        # Последний чанк
        if buffer:
            filename = os.path.join(output_dir, f"chunk_{file_counter:04d}.csv")
            pd.DataFrame(buffer).to_csv(filename, index=False)
            print(f"💾 Сохранено (остаток): {filename}")

    print(f"✅ Готово: создано {file_counter - 1} CSV-файлов.")


# === ЗАПУСК ===
if __name__ == "__main__":
    fits_file_path = r'D:\Code\Space_Warps\spAll-v6_1_3.fits'
    output_dir = r'D:\Code\Space_canvas\data\raw_data'

    if not os.path.exists(fits_file_path):
        print(f"❌ Ошибка: файл '{fits_file_path}' не найден.")
    else:
        lazy_create_csv_chunks(
            fits_file_path,
            chunk_size=1500,
            output_dir=output_dir
        )