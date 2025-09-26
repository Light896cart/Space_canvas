# logs/logger_config.py

import logging
import os
from pathlib import Path

# Определяем корень проекта (на уровень выше, чем этот файл)
project_root = Path(__file__).parent.parent

# Создаём папку logs, если её нет
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)

# Путь к файлу логов
log_file = log_dir / "app.log"

# Настройка базового логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),  # вывод в консоль
        logging.FileHandler(log_file, encoding='utf-8'),  # запись в файл
    ],
    force=True  # перезатирает предыдущие настройки (если уже были)
)

# Экспортируем основной логгер
logger = logging.getLogger(__name__.split(".")[0])  # например, 'logs'