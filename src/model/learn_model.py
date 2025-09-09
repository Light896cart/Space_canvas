from typing import List


import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader


from src.model.model_architecture import BaseModel


def learn_model(train_dataset):
    model = BaseModel()
    # --- ⚙️ Оптимизатор и лосс ---
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    # --- 🔁 Цикл обучения ---
    model.train()
    for epoch in range(1):
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