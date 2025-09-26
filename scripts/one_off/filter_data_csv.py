import pandas as pd
import os

def filter_sdss_spectra(
    input_csv,
    output_csv=None,
    min_sn_median_r=10.0,
    exclude_zwarning=True,
    required_columns=None
):
    """
    Фильтрует SDSS спектроскопические данные:
    - Исключает объекты с zwarning != 0
    - Оставляет только те, у кого sn_median_r >= min_sn_median_r
    - Конвертирует ra/dec в градусы (если в формате HH:MM:SS / ±DD:MM:SS)
    - Сохраняет очищенный датасет.

    Args:
        input_csv (str): Путь к исходному CSV.
        output_csv (str): Путь к выходному файлу. Если None — добавляет '_clean'.
        min_sn_median_r (float): Минимальное значение SNR (по умолчанию 10).
        exclude_zwarning (bool): Если True — оставляет только zwarning == 0.
        required_columns (list): Дополнительные колонки, которые должны быть.
    """

    # Авто-имя выходного файла
    if output_csv is None:
        name, ext = os.path.splitext(input_csv)
        output_csv = f"{name}_clean{ext}"

    # Проверка существования входного файла
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Файл не найден: {input_csv}")

    # Чтение CSV
    print(f"🔄 Читаем файл: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать CSV: {e}")

    initial_count = len(df)
    print(f"✅ Загружено {initial_count} строк")

    # Проверка обязательных колонок
    required = ['ra', 'dec']
    if exclude_zwarning:
        required.append('zwarning')
    if min_sn_median_r is not None:
        required.append('sn_median_r')

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют колонки: {missing}")

    # Конвертация RA/Dec, если они в строковом формате
    if df['ra'].dtype == 'object':
        print("🔄 Конвертируем ra (HH:MM:SS) → градусы...")
        df['ra'] = df['ra'].apply(hms_to_degrees)
    else:
        print("ℹ️ Колонка 'ra' уже в градусах — пропускаем конвертацию.")

    if df['dec'].dtype == 'object':
        print("🔄 Конвертируем dec (±DD:MM:SS) → градусы...")
        df['dec'] = df['dec'].apply(dms_to_degrees)
    else:
        print("ℹ️ Колонка 'dec' уже в градусах — пропускаем конвертацию.")

    # Фильтрация по zwarning
    if exclude_zwarning:
        mask = (df['zwarning'] == 0)
        filtered_count = mask.sum()
        print(f"🔍 Фильтруем по zwarning == 0: {filtered_count} из {initial_count}")
        df = df[mask].copy()

    # Фильтрация по sn_median_r
    if min_sn_median_r is not None and 'sn_median_r' in df.columns:
        valid_sn = df['sn_median_r'].notna() & (df['sn_median_r'] >= min_sn_median_r)
        sn_filtered = valid_sn.sum()
        print(f"🔍 Фильтруем по sn_median_r >= {min_sn_median_r}: {sn_filtered} объектов")
        df = df[valid_sn].copy()

    # Удаление строк с NaN в ra/dec после конвертации
    nan_coords = df[['ra', 'dec']].isna().any(axis=1).sum()
    if nan_coords > 0:
        print(f"🗑️ Удаляем {nan_coords} строк с некорректными координатами")
        df = df.dropna(subset=['ra', 'dec'])

    final_count = len(df)
    print(f"✅ Окончательный размер датасета: {final_count} строк ({final_count / initial_count:.1%})")

    # Сохранение
    df.to_csv(output_csv, index=False)
    print(f"💾 Очищенные данные сохранены в: {output_csv}")

    return df


# ———————————————————————————————————
# 🛠️ Вспомогательные функции (из твоего кода)
# ———————————————————————————————————

def hms_to_degrees(hms_str):
    """Преобразует RA (HH:MM:SS) в градусы."""
    try:
        h, m, s = map(float, str(hms_str).strip().split(':'))
        return (h + m / 60 + s / 3600) * 15  # 15 градусов на час
    except Exception as e:
        print(f"⚠️ Ошибка при парсинге RA: {hms_str} → {e}")
        return float('nan')


def dms_to_degrees(dms_str):
    """Преобразует Dec (±DD:MM:SS) в градусы."""
    try:
        raw = str(dms_str).strip()
        sign = -1 if raw.startswith('-') else 1
        dms_clean = raw.replace('-', '', 1).replace('+', '', 1) if raw[0] in ['-', '+'] else raw
        d, m, s = map(float, dms_clean.split(':'))
        return sign * (d + m / 60 + s / 3600)
    except Exception as e:
        print(f"⚠️ Ошибка при парсинге Dec: {dms_str} → {e}")
        return float('nan')


# ———————————————————————————————————
# 🚀 Пример использования
# ———————————————————————————————————

if __name__ == "__main__":
    filter_sdss_spectra(
        input_csv=r"D:\Code\Space_canvas\data_test\optical_search_699065.csv",
        output_csv=r"D:\Code\Space_canvas\data_test\new.csv",
        min_sn_median_r=0.0,
        exclude_zwarning=True
    )