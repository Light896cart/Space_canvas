import requests
from astropy.table import Table
from astropy.io import fits
from astropy.visualization import AsinhStretch, PercentileInterval
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import warnings

# Отключаем предупреждения для скорости
warnings.filterwarnings('ignore')

# Глобальная сессия для переиспользования соединений
_session = None


def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    return _session


def get_cutout_url(ra, dec, filter_band='i'):
    """Оптимизированное получение URL"""
    try:
        url_table = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filter_band}"
        table = Table.read(url_table, format='ascii')

        # Векторизованная фильтрация
        mask = table['filter'] == filter_band
        if not np.any(mask):
            return None

        filename = table['filename'][mask][0]
        url = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size=75&format=fits&red={filename}"
        )
        return url
    except:
        return None


def download_image(url, size=75):
    """Оптимизированная загрузка изображения"""
    try:
        # Быстрая замена размера
        if "size=75" in url:
            url = url.replace("size=75", f"size={size}")

        session = get_session()
        response = session.get(url, timeout=12)
        response.raise_for_status()

        # Быстрая обработка FITS
        with fits.open(BytesIO(response.content), memmap=False, lazy_load_hdus=False) as hdul:
            data = hdul[0].data

        if data is None:
            return None

        # Оптимизированная обработка контраста
        # Используем более быстрые перцентили
        p995 = np.percentile(data, 99.5)
        p005 = np.percentile(data, 0.5)

        # Векторизованная нормализация
        normalized = np.clip((data - p005) / (p995 - p005 + 1e-8), 0, 1)

        # Быстрый asinh stretch
        stretched = np.arcsinh(normalized * 3) / np.arcsinh(3)

        return stretched.astype(np.float32)

    except Exception:
        return None


def get_ps1_multiband(ra, dec, filters=('g', 'r', 'i', 'z', 'y'), size=75):
    """
    Максимально быстрая рабочая версия
    """
    # Предварительная валидация
    try:
        ra_float, dec_float = float(ra), float(dec)
        if not (-360 <= ra_float <= 360 and -90 <= dec_float <= 90):
            return None
    except:
        return None

    # Параллельное получение всех URL
    urls = {}
    url_params = [(band, ra, dec, band) for band in filters]

    with ThreadPoolExecutor(max_workers=len(filters)) as executor:
        # Получаем все URL параллельно
        future_to_band = {
            executor.submit(get_cutout_url, ra, dec, band): band
            for band in filters
        }

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                url = future.result()
                if url is None:
                    return None
                urls[band] = url
            except:
                return None

    # Параллельная загрузка всех изображений
    images = {}

    with ThreadPoolExecutor(max_workers=len(filters)) as executor:
        future_to_band = {
            executor.submit(download_image, urls[band], size): band
            for band in filters
        }

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                img = future.result()
                if img is None:
                    return None
                images[band] = img
            except:
                return None

    # Проверка и сборка куба
    if len(images) != len(filters):
        return None

    try:
        # Быстрая проверка размеров
        shapes = [img.shape for img in images.values()]
        if len(set(shapes)) > 1:  # Все формы должны быть одинаковыми
            return None

        base_shape = shapes[0]

        # Быстрая сборка куба с preallocated array
        cube = np.empty((*base_shape, len(filters)), dtype=np.float32)
        for i, band in enumerate(filters):
            cube[..., i] = images[band]

        return cube

    except Exception:
        return None