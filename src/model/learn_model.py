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
        for epoch in range(1):
            for batch in train_dataset:
                images, labels = batch
                labels = labels[:, 0]  # ← ВАЖНО: (N, 1) → (N,)

                optimizer.zero_grad()
                outputs = model(images)  # ← ДОЛЖНО БЫТЬ: (N, 3)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                print(f"Loss: {loss.item()}")