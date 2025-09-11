import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def compute_batch_metrics(labels, preds, probs=None, prefix="train"):
    """
    Вычисляет метрики для одного батча.

    Args:
        labels (Tensor): истинные метки, shape (N,)
        preds (Tensor): предсказанные метки, shape (N,)
        probs (Tensor, optional): вероятности после softmax, shape (N, C)
        prefix (str): префикс, например 'train', 'val'

    Returns:
        dict: метрики с префиксом, готовые к wandb.log()
    """
    # Переводим в numpy для sklearn
    y_true = labels.cpu().numpy()
    y_pred = preds.cpu().numpy()

    metrics = {}

    # Accuracy
    metrics[f"{prefix}_acc"] = accuracy_score(y_true, y_pred)

    # Другие метрики (macro — устойчиво к дисбалансу)
    metrics[f"{prefix}_prec"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f"{prefix}_rec"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics[f"{prefix}_f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted (если важны частоты классов)
    metrics[f"{prefix}_f1_weighted"] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Средняя уверенность (confidence)
    if probs is not None:
        conf = probs.max(dim=1)[0].mean().item()  # среднее по максимальным вероятностям
        metrics[f"{prefix}_conf"] = conf

    # Loss будет добавлен отдельно — но можно и здесь зарезервировать
    return metrics