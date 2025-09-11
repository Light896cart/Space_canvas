import time
import matplotlib.pyplot as plt
import numpy as np
import requests
from astropy.io import fits
from astropy.table import Table
from astropy.visualization import PercentileInterval, AsinhStretch
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential

# === Функция для получения URL через таблицу (обязательно) ===
def get_cutout_url(ra, dec, filter_band='i'):
    """Получает URL на FITS-изображение для заданного RA, Dec и фильтра"""
    try:
        # Запрос к ps1filenames.py
        url_table = f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters={filter_band}"
        table = Table.read(url_table, format='ascii')

        # Фильтруем по нужному фильтру
        table = table[table['filter'] == filter_band]
        if len(table) == 0:
            return None

        filename = table['filename'][0]  # берём первый файл
        # Формируем URL для fitscut
        url = (
            f"https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
            f"ra={ra}&dec={dec}&size=75&format=fits&red={filename}"
        )
        return url
    except Exception as e:
        print(f"Ошибка при получении URL ({ra}, {dec}): {e}")
        return None

# === Загрузка изображения по URL ===
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, max=10),
    reraise=True
)
def download_image(url):
    """Загружает FITS по URL и возвращает обработанное изображение"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with fits.open(BytesIO(response.content)) as hdul:
            data = hdul[0].data
        # Улучшаем контраст
        transform = AsinhStretch() + PercentileInterval(99.5)
        return transform(data)
    except Exception as e:
        print('Вот такая вот ошибка',e)
        raise  # Чтобы retry попробовал снова


# === Основная функция: получить изображение ===
def get_ps1_image(ra, dec, filter_band='i'):
    url = get_cutout_url(ra, dec, filter_band)
    if url is None:
        return None
    return download_image(url)


# # Старые координаты (ваш первоначальный список)
# ra_list_old = [35.103846, 35.01667, 34.90911, 34.972713, 34.991105,
#                35.056275, 35.407754, 35.198357, 35.410434, 35.689222,
#                35.156116, 35.164636, 35.24326, 35.337295, 35.266138,
#                35.130754, 35.369911, 35.617758, 35.61671, 35.417946,
#                35.652879, 35.95899, 36.048021, 36.13731, 36.289988,
#                36.190746, 36.246152, 35.820523, 35.810521, 35.891618,
#                35.838609]
#
# dec_list_old = [-6.2921282, -6.2686349, -6.0668356, -6.1898776, -6.0930068,
#                 -6.0171515, -6.4484517, -6.2015798, -6.2627868, -6.2771255,
#                 -6.1768468, -6.0885777, -6.0288194, -6.1206512, -6.04728,
#                 -5.8761376, -5.9827167, -6.0965984, -6.0234278, -5.8505189,
#                 -5.898463, -6.2793123, -6.1544588, -6.2008188, -6.2221293,
#                 -6.1972574, -6.1384437, -6.0747067, -6.0036851, -6.0810754,
#                 -5.9334336]
#
# # Параллельная загрузка
# print("Загружаем 10 изображений...")
# start = time.time()
#
# with ThreadPoolExecutor(max_workers=5) as executor:
#     args = [(ra_list_old[i], dec_list_old[i], 'i') for i in range(len(dec_list_old))]
#     images = list(executor.map(lambda p: get_ps1_image(*p), args))
#
# total_time = time.time() - start
# print(f"✅ Загружено за {total_time:.2f} сек (среднее {total_time / 10:.2f} с/изобр)")
#
# # Отображение
# fig, axes = plt.subplots(7, 5, figsize=(15, 6))
# axes = axes.ravel()
#
# for i, img in enumerate(images):
#     if img is not None:
#         axes[i].imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1)
#         axes[i].set_title(f"#{i + 1}", fontsize=8)
#     else:
#         axes[i].set_title(f"❌ #{i + 1}", color='red', fontsize=8)
#     axes[i].axis('off')
#
# plt.suptitle("10 изображений с PS1 — исправленная загрузка", y=1.02, fontsize=16)
# plt.tight_layout()
# plt.show()