from typing import List

import wandb
from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.dataloader import create_train_val_dataloader
from src.metrics.classification import compute_batch_metrics
from src.model.model_architecture import BaseModel


def eval_model(
        model,
        val_dataset,
):
    # --- 🟢 Валидация ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataset:
            images, labels = batch
            labels = labels[:, 0]
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            val_acc = correct / total
            print('ЭТО ВАЛ АССЕС', val_acc)


def train_model(
        folder: str,
        list_label: list[str],
        path_img: str,
        train_ratio: int | None = 0.9,
        list_extra: list[str] | None = None,
        transform: transforms.Compose | None = None,
        path_val_dataset: str | None = None
):
    """
    Обучение нейронной сети

    Args:
        folder: Путь до папки с csv файлами (до датасета)
        list_label: Список столбцов для меток
        path_img: Путь до папки с изображениями
        train_ratio: Сколько должно быть тренировочных данных в процентах
        list_extra: Дополнительные экстра параметры которые будут внедрять в модель вместе с изображением
        transform: Трансформация изображения (изменение изображения)
        path_val_dataset: Если есть путь до csv файла c валидационными данными можно указать

    Return:
        None
    """
    model = BaseModel()
    folder = Path(folder)

    pattern = "chunk_*.csv"
    # Получаем файлы
    files = sorted(folder.glob(pattern))
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_wts = None
    weight_model_new = None
    bias_model_new = None
    # --- ⚙️ Оптимизатор и лосс ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    val_dataset = None
    try:
        # --- 🔁 Цикл обучения ---
        model.train()
        for epoch in range(1):
            for path_csv in files:
                train_dataset, val_dataset = create_train_val_dataloader(
                    path_csv=path_csv,
                    path_img=path_img,
                    list_extra=list_extra,
                    list_label=list_label,
                    train_ratio=train_ratio,
                    transform=transform,
                    path_val_dataset=path_val_dataset
                )
                val_dataset = val_dataset
                running_loss = 0.0
                progress_bar = tqdm(
                    train_dataset,
                    desc=f"Epoch {epoch + 1}/{epoch}",
                    unit="batch",
                    disable=not True,
                    leave=False
                )
                for step, batch in enumerate(progress_bar):
                    images, labels = batch
                    labels = labels[:, 0]  # ← ВАЖНО: (N, 1) → (N,)

                    optimizer.zero_grad()
                    outputs = model(images)  # ← ДОЛЖНО БЫТЬ: (N, 3)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # Предсказания и вероятности
                    preds = outputs.argmax(dim=1)
                    probs = torch.softmax(outputs, dim=1)

                    # Вычисляем метрики по батчу
                    batch_metrics = compute_batch_metrics(labels, preds, probs, prefix="train")

                    # Добавляем loss
                    batch_metrics["train_loss"] = loss.item()
                    print('Обучение')
                    # Добавляем номер шага (опционально)
                    batch_metrics["step"] = epoch * len(train_dataset) + step

                    # 🚀 Отправляем ВСЁ в W&B
                    wandb.log(batch_metrics, commit=True)

                    # 🖨 Опционально: вывод в консоль
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "acc": f"{batch_metrics['train_acc']:.3f}",
                        "f1": f"{batch_metrics['train_f1']:.3f}"
                    })
    except KeyboardInterrupt:
        wandb.finish()
        eval_model(model,val_dataset)
