import json
import random
import sys
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from dataset import KvasirDataset
from metrics import score_masks

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchKvasirDataset(Dataset):
    def __init__(self, dataset_dir, split, csv_path="/content/splits.csv", augment=False):
        self.base = KvasirDataset(dataset_dir, split, csv_path=csv_path, augment=augment)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        image, mask = self.base[index]
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


class PerImageSoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits.float())
        b = probs.size(0)
        probs_flat = probs.view(b, -1)
        targets_flat = targets.float().view(b, -1)
        inter = (probs_flat * targets_flat).sum(dim=1)
        denom = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, loss_mode: str = "bce_dice", bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.loss_mode = loss_mode.lower()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = PerImageSoftDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        if self.loss_mode == "bce":
            return bce_loss
        return self.bce_weight * bce_loss + self.dice_weight * self.dice(logits, targets)


def build_unet():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


@torch.inference_mode()
def evaluate_split(model, loader, device, criterion=None, threshold=0.5, use_amp=False):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    predictions, targets = [], []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks) if criterion is not None else None

        probs = torch.sigmoid(logits.float()).squeeze(1).cpu().numpy()
        predictions.extend((probs >= threshold).astype(np.uint8))
        targets.extend(masks.squeeze(1).cpu().numpy().astype(np.uint8))

        bs = images.size(0)
        if loss is not None:
            total_loss += loss.item() * bs
        num_samples += bs

    scores = score_masks(predictions, targets)
    out = {"mean_iou": scores["mean_iou"], "mean_dice": scores["mean_dice"]}
    if criterion is not None:
        out["loss"] = total_loss / max(num_samples, 1)
    return out


def run_training(
    dataset_dir="/content/drive/MyDrive/Kvasir-SEG",
    csv_path="/content/splits.csv",
    save_dir="/content/drive/MyDrive/cv_project/checkpoints",
    batch_size=16,
    epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    threshold=0.5,
    seed=42,
    loss_mode="bce_dice",
    use_amp=True,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(use_amp and device.type == "cuda")
    print(f"실행 장치: {device} | AMP: {amp_enabled} | Loss: {loss_mode}")

    train_ds = TorchKvasirDataset(dataset_dir, "train", csv_path=csv_path, augment=True)
    val_ds = TorchKvasirDataset(dataset_dir, "val", csv_path=csv_path, augment=False)
    test_ds = TorchKvasirDataset(dataset_dir, "test", csv_path=csv_path, augment=False)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = build_unet().to(device)
    criterion = SegmentationLoss(loss_mode=loss_mode, bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_path / f"unet_resnet34_{loss_mode}_best.pth"
    history_path = save_path / f"unet_resnet34_{loss_mode}_history.json"

    history = []
    best_val_dice = -1.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0

        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = images.size(0)
            running_loss += loss.item() * bs
            processed += bs

        train_loss = running_loss / max(processed, 1)
        val_metrics = evaluate_split(model, val_loader, device, criterion=criterion, threshold=threshold, use_amp=amp_enabled)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["mean_iou"],
            "val_dice": val_metrics["mean_dice"],
            "lr": current_lr,
        })

        print(
            f"[Epoch {epoch:02d}/{epochs:02d}] train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_IoU={val_metrics['mean_iou']:.4f} | "
            f"val_Dice={val_metrics['mean_dice']:.4f}"
        )

        if val_metrics["mean_dice"] > best_val_dice:
            best_val_dice = val_metrics["mean_dice"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_iou": val_metrics["mean_iou"],
                    "val_dice": val_metrics["mean_dice"],
                    "threshold": threshold,
                },
                best_ckpt_path,
            )

    checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_split(model, test_loader, device, criterion=criterion, threshold=threshold, use_amp=amp_enabled)
    print(
        f"\n완료! Best Epoch {best_epoch} (Val Dice: {best_val_dice:.4f}) | "
        f"Test Mean IoU: {test_metrics['mean_iou']:.4f} | "
        f"Test Mean Dice: {test_metrics['mean_dice']:.4f}"
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "test_mean_iou": test_metrics["mean_iou"],
        "test_mean_dice": test_metrics["mean_dice"],
        "history": history,
    }
    history_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
