from typing import List

from torchvision import transforms
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.dataloader import create_train_val_dataloader
from src.model.model_architecture import BaseModel


def learn_model(
        folder: str,
        list_label: list[str],
        path_img: str,
        train_ratio: int | None = 0.9,
        list_extra: list[str] | None = None,
        transform: transforms.Compose | None = None,
        path_val_dataset: str | None = None
):
    model = BaseModel()
    folder = Path(folder)

    pattern = "spall_chunk_*.csv"
    # Получаем файлы
    files = sorted(folder.glob(pattern))
    print('патерн',files)
    train_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    best_model_wts = None
    weight_model_new = None
    bias_model_new = None
    # --- ⚙️ Оптимизатор и лосс ---
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
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
                for batch in progress_bar:
                    images, labels = batch
                    labels = labels[:, 0]  # ← ВАЖНО: (N, 1) → (N,)

                    optimizer.zero_grad()
                    outputs = model(images)  # ← ДОЛЖНО БЫТЬ: (N, 3)

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * images.size(0)
                    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

                    # 👇 Вычисляем accuracy для текущего батча
                    preds = outputs.argmax(dim=1)  # предсказанные классы
                    correct = (preds == labels).sum().item()  # число правильных
                    total = labels.size(0)  # размер батча
                    batch_acc = correct / total * 100.0

                    # 🖨 Выводим loss и accuracy для текущего батча
                    print(f"Loss: {loss.item():.4f} | Accuracy: {batch_acc:.2f}%")
    except KeyboardInterrupt:
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
                print('ЭТО ВАЛ АССЕС',val_acc)
            val_accuracies.append(val_acc)
            # --- 📢 Логирование ---
            print(f"Epoch Val Acc: {val_acc:6.4f} | Best: {best_val_acc:6.4f}")
