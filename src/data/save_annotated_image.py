import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm


# ========================
# 🧭 Вычисление ограничивающего прямоугольника (bounding box)
# ========================
def compute_bbox_from_center_fov(ra_center: float, dec_center: float, fov_deg: float, width: int, height: int) -> tuple:
    """
    Вычисляет координаты ограничивающего прямоугольника (bounding box) на небесной сфере
    по центру изображения, полю зрения и размерам изображения в пикселях.

    Параметры:
    ----------
    ra_center : float
        Прямое восхождение центра изображения (в градусах, [0, 360)).
    dec_center : float
        Склонение центра изображения (в градусах, [-90, +90]).
    fov_deg : float
        Поле зрения по большей стороне изображения (в градусах).
    width : int
        Ширина изображения в пикселях.
    height : int
        Высота изображения в пикселях.

    Возвращает:
    -----------
    tuple[float, float, float, float]
        (ra_min, ra_max, dec_min, dec_max) — координаты bbox в градусах.
        RA может «пересекать» 0° (например, ra_min = 359°, ra_max = 1°).
    """
    if width >= height:
        scale = fov_deg / width
        fov_y_deg = scale * height
    else:
        scale = fov_deg / height
        fov_y_deg = scale * width

    ra_min = ra_center - (fov_deg / 2)
    ra_max = ra_center + (fov_deg / 2)
    dec_min = dec_center - (fov_y_deg / 2)
    dec_max = dec_center + (fov_y_deg / 2)

    # Нормализация RA в диапазон [0, 360)
    ra_min = ra_min % 360
    ra_max = ra_max % 360

    return ra_min, ra_max, dec_min, dec_max


# ========================
# 🔍 Поиск объектов в ограничивающем прямоугольнике
# ========================
def find_objects_in_bbox(
    ra_center: float,
    dec_center: float,
    ra_min: float,
    ra_max: float,
    dec_min: float,
    dec_max: float,
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Находит объекты из DataFrame, попадающие в заданный bounding box на небесной сфере.
    Учитывает возможное пересечение RA = 0°.

    Параметры:
    ----------
    ra_center, dec_center : float
        Центр изображения (для точного совпадения при необходимости).
    ra_min, ra_max : float
        Минимальное и максимальное прямое восхождение bbox (в градусах).
    dec_min, dec_max : float
        Минимальное и максимальное склонение bbox (в градусах).
    df : pd.DataFrame
        Таблица с колонками 'ra' и 'dec' (в градусах).

    Возвращает:
    -----------
    pd.DataFrame
        Подмножество df, содержащее только объекты внутри bbox.
        Если ни один объект не попал в bbox, но центральный объект есть — возвращает его.
    """
    tolerance = 1e-5
    mask_center = (
        (abs(df['ra'] - ra_center) < tolerance) &
        (abs(df['dec'] - dec_center) < tolerance)
    )
    center_objects = df[mask_center]

    # Фильтр по склонению (Dec)
    mask_dec = (df['dec'] >= dec_min) & (df['dec'] <= dec_max)

    # Фильтр по прямому восхождению (RA) — с учётом пересечения 0°
    if ra_min <= ra_max:
        mask_ra = (df['ra'] >= ra_min) & (df['ra'] <= ra_max)
    else:
        mask_ra = (df['ra'] >= ra_min) | (df['ra'] <= ra_max)

    mask = mask_ra & mask_dec
    objects_in_bbox = df[mask]

    # Если центральный объект есть, но он не попал в bbox (например, из-за края),
    # добавляем его вручную
    if len(center_objects) > 0 and len(objects_in_bbox) == 0:
        objects_in_bbox = center_objects

    return objects_in_bbox


# ========================
# 🎯 Конвертация небесных координат → нормализованные пиксельные координаты
# ========================
def convert_to_normalized_point(
    ra_obj: float,
    dec_obj: float,
    ra_center: float,
    dec_center: float,
    fov_deg: float,
    img_width: int,
    img_height: int
) -> tuple:
    """
    Преобразует небесные координаты объекта (RA, Dec) в нормализованные пиксельные координаты (x, y),
    где (0,0) — верхний левый угол, (1,1) — нижний правый угол изображения.

    Параметры:
    ----------
    ra_obj, dec_obj : float
        Координаты объекта (в градусах).
    ra_center, dec_center : float
        Центр изображения (в градусах).
    fov_deg : float
        Поле зрения по большей стороне изображения (в градусах).
    img_width, img_height : int
        Размеры изображения в пикселях.

    Возвращает:
    -----------
    tuple[float, float]
        Нормализованные координаты (x_norm, y_norm) ∈ [0, 1].
    """
    if img_width >= img_height:
        scale = fov_deg / img_width
    else:
        scale = fov_deg / img_height

    # Разница в координатах
    delta_ra = ra_obj - ra_center
    delta_dec = dec_obj - dec_center

    # Пиксельные координаты относительно центра
    x_px = img_width / 2 + (delta_ra / scale)
    y_px = img_height / 2 - (delta_dec / scale)  # минус — ось Dec направлена вверх

    # Нормализация
    x_norm = x_px / img_width
    y_norm = y_px / img_height

    # Ограничение значений в [0, 1] (на случай выхода за край изображения)
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    return x_norm, y_norm


# ========================
# 🖼️ Получение цветного изображения с сервера Aladin (HiPS → FITS/JPG)
# ========================
def get_sdss9_color(
    ra: float,
    dec: float,
    fov_deg: float = 0.15,
    width: int = 512,
    height: int = 512,
    fmt: str = 'jpg'
) -> Image.Image | None:
    """
    Запрашивает цветное астрономическое изображение из сервиса Aladin HiPS (DESI Legacy Surveys DR10)
    по заданным координатам и параметрам.

    Параметры:
    ----------
    ra, dec : float
        Центр изображения (в градусах).
    fov_deg : float, optional
        Поле зрения по большей стороне (в градусах). По умолчанию 0.15.
    width, height : int, optional
        Размеры изображения в пикселях. По умолчанию 512×512.
    fmt : str, optional
        Формат изображения: 'jpg', 'png' и т.д. По умолчанию 'jpg'.

    Возвращает:
    -----------
    PIL.Image.Image | None
        Изображение в формате PIL, или None в случае ошибки.
    """
    url = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
    params = {
        'hips': 'CDS/P/DESI-Legacy-Surveys/DR10/color',
        'ra': ra,
        'dec': dec,
        'width': width,
        'height': height,
        'fov': fov_deg,
        'projection': 'TAN',
        'format': fmt
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Referer': 'https://aladin.cds.unistra.fr/'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        if response.status_code == 200:
            if fmt in ('jpeg', 'jpg', 'png'):
                img = Image.open(BytesIO(response.content))
                return img
            else:
                return response.content
        else:
            print(f"❌ Ошибка при запросе изображения: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Исключение при запросе изображения: {e}")
        return None


# ========================
# 🚀 Основной пайплайн: генерация изображений и аннотаций без дублирования объектов
# ========================
def generate_sdss_image_dataset(
        df: pd.DataFrame,
):
    """
    Основная функция пайплайна:
    1. Загружает CSV с координатами объектов.
    2. Для каждого ещё не обработанного объекта запрашивает изображение.
    3. Находит все объекты, попавшие на это изображение.
    4. Преобразует их координаты в нормализованные (x, y).
    5. Сохраняет изображение (.jpg) и аннотации (.csv).
    6. Удаляет обработанные объекты из DataFrame, чтобы избежать дубликатов.

    Конфигурация:
    -------------
    CSV_PATH : str
        Путь к входному CSV-файлу с колонками 'ra', 'dec'.
    OUTPUT_DIR : str
        Директория для сохранения изображений и аннотаций.
    """

    first_row = df.iloc[0]
    ra = first_row['ra']
    dec = first_row['dec']

    # Получаем изображение
    img = get_sdss9_color(ra, dec, fov_deg=0.05, width=512, height=512, fmt='jpg')
    if img is None:
        print(f"⚠️  Не удалось получить изображение для RA={ra}, Dec={dec}. Пропускаем.")
        df = df.iloc[1:].reset_index(drop=True)
        return

    # Вычисляем bbox
    ra_min, ra_max, dec_min, dec_max = compute_bbox_from_center_fov(ra, dec, 0.05, 512, 512)

    # Находим все объекты на этом изображении
    objects_on_image = find_objects_in_bbox(ra, dec, ra_min, ra_max, dec_min, dec_max, df)
    if len(objects_on_image) == 0:
        print(f"⚠️  Нет объектов на изображении для RA={ra}, Dec={dec}. Пропускаем.")
        df = df.iloc[1:].reset_index(drop=True)
        return

    # Добавляем нормализованные координаты
    objects_on_image = objects_on_image.copy()
    normalized_points = objects_on_image.apply(
        lambda row: convert_to_normalized_point(
            ra_obj=row['ra'],
            dec_obj=row['dec'],
            ra_center=ra,
            dec_center=dec,
            fov_deg=0.05,
            img_width=512,
            img_height=512
        ),
        axis=1
    )
    objects_on_image[['x_center_norm', 'y_center_norm']] = pd.DataFrame(
        normalized_points.tolist(), index=objects_on_image.index
    )

    return img,objects_on_image
