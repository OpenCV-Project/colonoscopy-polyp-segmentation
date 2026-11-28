import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

from config import load_config
import train_unet


def plot_curves(history_path, save_img_path):
    data = json.loads(Path(history_path).read_text(encoding="utf-8"))
    hist = data["history"]
    epochs = [h["epoch"] for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    axes[0].plot(epochs, [h["train_loss"] for h in hist], label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, [h["val_loss"] for h in hist], label="Val Loss", marker="s", markersize=3)
    axes[0].axvline(data["best_epoch"], color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({data['best_epoch']})")
    axes[0].set_title("U-Net (ResNet-34) Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, [h["val_dice"] for h in hist], label="Val Dice", color="tab:green", marker="o", markersize=3)
    axes[1].plot(epochs, [h["val_iou"] for h in hist], label="Val IoU", color="tab:orange", marker="s", markersize=3)
    axes[1].axvline(data["best_epoch"], color="gray", linestyle="--", alpha=0.7, label=f"Best Epoch ({data['best_epoch']})")
    axes[1].set_title(f"Validation Metrics (Test Dice: {data['test_mean_dice']:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_img_path)
    plt.close(fig)


def plot_predictions(cfg, ckpt_path, save_img_path, sample_indices=None):
    if sample_indices is None:
        sample_indices = [0, 15, 30, 60, 90, 120]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_unet.build_unet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = train_unet.TorchKvasirDataset(
        cfg["data"]["dataset_dir"], "test", csv_path=cfg["data"]["csv_path"], augment=False
    )
    threshold = cfg["model"]["threshold"]

    fig, axes = plt.subplots(len(sample_indices), 4, figsize=(12, 3 * len(sample_indices)), dpi=150)
    col_titles = ["Input Image", "Ground Truth", "U-Net Prediction", "Overlay (GT:Green / Pred:Red)"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            img_t, mask_t = test_ds[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits).squeeze().cpu().numpy() >= threshold).astype(np.uint8)
            gt = mask_t.squeeze().numpy().astype(np.uint8)

            img_vis = np.clip(img_t.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            overlay = img_vis.copy()
            overlay[gt == 1] = overlay[gt == 1] * 0.5 + np.array([0.0, 0.8, 0.0]) * 0.5
            overlay[pred == 1] = overlay[pred == 1] * 0.5 + np.array([0.9, 0.1, 0.1]) * 0.5

            inter = np.logical_and(pred, gt).sum()
            denom = pred.sum() + gt.sum()
            dice = (2.0 * inter + 1e-7) / (denom + 1e-7)

            imgs = [img_vis, gt, pred, overlay]
            for col in range(4):
                ax = axes[row, col]
                ax.imshow(imgs[col], cmap="gray" if col in [1, 2] else None)
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
                if col == 2:
                    ax.set_xlabel(f"Sample #{idx} | Dice: {dice:.4f}", fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_img_path)
    plt.close(fig)


if __name__ == "__main__":
    cfg = load_config()
    save_dir = Path(cfg["data"]["save_dir"])
    loss_mode = cfg["train"]["loss_mode"]
    history_path = save_dir / f"unet_resnet34_{loss_mode}_history.json"
    ckpt_path = save_dir / f"unet_resnet34_{loss_mode}_best.pth"

    plot_curves(history_path, save_dir / "unet_training_curves.png")
    plot_predictions(cfg, ckpt_path, save_dir / "unet_test_predictions.png")
    print("평가 및 시각화 완료!")
