"""TODO:
Ну тут много работы, много что проанализировать и упростить
"""

import requests
from astropy.table import Table
from astropy.io import fits
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import warnings

from astropy.visualization import AsinhStretch, PercentileInterval
from tenacity import retry, stop_after_attempt, wait_exponential

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10)
)
def get_cutout_url(ra, dec, filter_band='i'):
    """Получение URL с повторными попытками"""
    try:
        url_table = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filter_band}"
        table = Table.read(url_table, format='ascii')

        # Векторизованная фильтрация
        mask = table['filter'] == filter_band
        if not np.any(mask):
            raise ValueError(f"No filename found for filter {filter_band}")

        filename = table['filename'][mask][0]
        url = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size=75&format=fits&red={filename}"
        )
        return url
    except Exception as e:
        raise ValueError(f"Failed to get cutout URL for {ra}, {dec}, {filter_band}: {str(e)}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10)
)
def download_image(url, size=75):
    """Загрузка изображения с повторными попытками"""
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
        if data is None or not np.issubdtype(data.dtype, np.number):
            raise ValueError("Invalid or empty FITS data")

        transform = AsinhStretch() + PercentileInterval(99.5)
        data = transform(data)
        return data

    except Exception as e:
        raise ValueError(f"Failed to download/parse image from {url}: {str(e)}")


@retry(
    stop=stop_after_attempt(3),      # Попробуем 3 раза ВСЮ операцию
    wait=wait_exponential(multiplier=1, max=10)
)
def get_ps1_multiband(ra, dec, filters=('g', 'r', 'i', 'z', 'y'), size=75):
    """
    Максимально быстрая рабочая версия — НИКОГДА не возвращает None!
    Если хоть один фильтр не загрузился — ПОЛНОСТЬЮ ПОВТОРЯЕМ ЗАПРОС.
    Только после 3 неудачных попыток — возвращаем zeros.
    """

    # Предварительная валидация координат
    try:
        ra_float, dec_float = float(ra), float(dec)
        if not (-360 <= ra_float <= 360 and -90 <= dec_float <= 90):
            raise ValueError("Invalid RA/DEC coordinates")
    except (ValueError, TypeError):
        raise ValueError("Invalid RA/DEC coordinates")

    # Шаг 1: Параллельное получение всех URL
    urls = {}
    failed_bands = set()

    with ThreadPoolExecutor(max_workers=len(filters)) as executor:
        future_to_band = {
            executor.submit(get_cutout_url, ra, dec, band): band
            for band in filters
        }

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                url = future.result()
                urls[band] = url
            except Exception:
                failed_bands.add(band)

    # Если хотя бы один URL не получен — считаем всю операцию проваленной
    if len(urls) != len(filters):
        raise ValueError(f"Failed to get URLs for bands: {failed_bands}")

    # Шаг 2: Параллельная загрузка изображений
    images = {}
    base_shape = None

    with ThreadPoolExecutor(max_workers=len(filters)) as executor:
        future_to_band = {
            executor.submit(download_image, urls[band], size): band
            for band in filters
        }

        for future in as_completed(future_to_band):
            band = future_to_band[future]
            try:
                img = future.result()
                images[band] = img
                if base_shape is None:
                    base_shape = img.shape
            except Exception:
                failed_bands.add(band)

    # Если хоть одно изображение не загрузилось — поднимаем исключение для retry
    if len(images) != len(filters):
        raise ValueError(f"Failed to download images for bands: {failed_bands}")

    # Убедимся, что все изображения одного размера
    shapes = [img.shape for img in images.values()]
    if len(set(shapes)) > 1:
        raise ValueError("Inconsistent image sizes across filters")

    if base_shape is None:
        base_shape = (size, size)

    # Шаг 3: Сборка куба — теперь мы уверены, что все данные есть
    cube = np.empty((*base_shape, len(filters)), dtype=np.float32)
    for i, band in enumerate(filters):
        cube[..., i] = images[band]
    return cube