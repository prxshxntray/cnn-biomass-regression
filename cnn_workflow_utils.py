"""Shared utilities for CNN notebooks."""

from __future__ import annotations

import random
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset
from torchvision import models
from PIL import Image

TARGET_COLUMNS = [
    "Dry_Clover_g",
    "Dry_Dead_g",
    "Dry_Green_g",
    "Dry_Total_g",
    "GDM_g",
]


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the best available device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(
    architecture: str = "resnet18",
    num_targets: int = 5,
    weights: Optional[object] = None,
) -> nn.Module:
    """Create a torchvision model with a regression head."""
    architecture = architecture.lower()

    if architecture == "resnet18":
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_targets)
    elif architecture == "densenet121":
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_targets)
    elif architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_targets)
    else:
        raise ValueError(
            "Unknown architecture. Use 'resnet18', 'densenet121', or 'efficientnet_b0'."
        )

    return model


def create_kfold(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
) -> KFold:
    """Return a configured KFold splitter."""
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def scale_targets(
    dataframe,
    target_cols: Optional[Iterable[str]] = None,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[object, StandardScaler]:
    """Scale target columns and return the transformed dataframe and scaler."""
    if target_cols is None:
        target_cols = TARGET_COLUMNS

    scaler = scaler or StandardScaler()
    dataframe = dataframe.copy()
    dataframe[list(target_cols)] = scaler.fit_transform(dataframe[list(target_cols)])
    return dataframe, scaler


def mixup(images: torch.Tensor, targets: torch.Tensor, alpha: float = 0.4):
    """Apply mixup augmentation to a batch."""
    if alpha <= 0:
        return images, targets

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_targets = lam * targets + (1 - lam) * targets[index]

    return mixed_images, mixed_targets


class ImageDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_columns=None):
        self.dataframe = dataframe
        self.transform = transform
        self.target_columns = list(target_columns or TARGET_COLUMNS)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index: int):
        img_path = self.dataframe.iloc[index]["full_path"]
        image = Image.open(img_path).convert("RGB")

        targets = self.dataframe.iloc[index][self.target_columns].values.astype("float")
        targets = torch.tensor(targets, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, targets


def train_regression(
    model: nn.Module,
    num_epochs: int,
    train_dl,
    valid_dl,
    loss_fn,
    optimizer,
    device: torch.device,
    mixup_fn=None,
):
    """Train a regression model and return loss histories."""
    loss_hist_train = [0.0] * num_epochs
    loss_hist_valid = [0.0] * num_epochs

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            if mixup_fn is not None:
                x_batch, y_batch = mixup_fn(x_batch, y_batch)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_hist_train[epoch] += loss.item()

        loss_hist_train[epoch] /= len(train_dl)

        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item()

        loss_hist_valid[epoch] /= len(valid_dl)

        print(
            f"Epoch {epoch + 1:2d}/{num_epochs} | "
            f"Train Loss: {loss_hist_train[epoch]:.6f} | "
            f"Val Loss: {loss_hist_valid[epoch]:.6f}"
        )

    return loss_hist_train, loss_hist_valid


def calculate_global_weighted_r2(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate globally weighted R² over all (image, target) pairs."""
    if preds.device != torch.device("cpu"):
        preds = preds.cpu()
    if targets.device != torch.device("cpu"):
        targets = targets.cpu()

    preds_np = preds.numpy()
    targets_np = targets.numpy()

    target_weights = {
        "Dry_Clover_g": 0.1,
        "Dry_Dead_g": 0.1,
        "Dry_Green_g": 0.1,
        "Dry_Total_g": 0.5,
        "GDM_g": 0.2,
    }

    n_samples = len(targets_np)
    flat_targets = targets_np.flatten()
    flat_preds = preds_np.flatten()

    weights = np.array(
        [
            target_weights["Dry_Clover_g"],
            target_weights["Dry_Dead_g"],
            target_weights["Dry_Green_g"],
            target_weights["Dry_Total_g"],
            target_weights["GDM_g"],
        ]
    )
    flat_weights = np.tile(weights, n_samples)

    y_bar_w = np.average(flat_targets, weights=flat_weights)
    ss_res = np.sum(flat_weights * (flat_targets - flat_preds) ** 2)
    ss_tot = np.sum(flat_weights * (flat_targets - y_bar_w) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)
