import csv
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


def save_training_curves(history_path, save_path):
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
    plt.savefig(save_path)
    plt.close(fig)


def plot_case_grid(records, title_prefix, save_path):
    n = len(records)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n), dpi=150)
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
    col_titles = ["Input Image", "Ground Truth", "U-Net Prediction", "Overlay (GT:Green / Pred:Red)"]

    for row, rec in enumerate(records):
        imgs = [rec["img_vis"], rec["gt"], rec["pred"], rec["overlay"]]
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap="gray" if col in [1, 2] else None)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
            if col == 2:
                ax.set_xlabel(f"{title_prefix} #{rec['idx']} | Dice: {rec['dice']:.4f} | IoU: {rec['iou']:.4f}", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def evaluate_and_export(cfg, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(cfg["data"]["save_dir"])
    loss_mode = cfg["train"]["loss_mode"]
    ckpt_path = ckpt_dir / f"unet_resnet34_{loss_mode}_best.pth"
    history_path = ckpt_dir / f"unet_resnet34_{loss_mode}_history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_unet.build_unet().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = train_unet.TorchKvasirDataset(
        cfg["data"]["dataset_dir"], "test", csv_path=cfg["data"]["csv_path"], augment=False
    )
    threshold = cfg["model"]["threshold"]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    records = []
    with torch.no_grad():
        for idx in range(len(test_ds)):
            img_t, mask_t = test_ds[idx]
            logits = model(img_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits).squeeze().cpu().numpy() >= threshold).astype(np.uint8)
            gt = mask_t.squeeze().numpy().astype(np.uint8)

            inter = float(np.logical_and(pred, gt).sum())
            union = float(np.logical_or(pred, gt).sum())
            denom = float(pred.sum() + gt.sum())

            dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
            iou = (inter + 1e-7) / (union + 1e-7)

            img_vis = np.clip(img_t.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            overlay = img_vis.copy()
            overlay[gt == 1] = overlay[gt == 1] * 0.5 + np.array([0.0, 0.8, 0.0]) * 0.5
            overlay[pred == 1] = overlay[pred == 1] * 0.5 + np.array([0.9, 0.1, 0.1]) * 0.5

            records.append({
                "idx": idx,
                "dice": dice,
                "iou": iou,
                "img_vis": img_vis,
                "gt": gt,
                "pred": pred,
                "overlay": overlay,
            })

    # 이미지별 점수 CSV 저장
    csv_out = out_dir / "test_per_image_scores.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "dice", "iou"])
        for r in records:
            writer.writerow([r["idx"], f"{r['dice']:.6f}", f"{r['iou']:.6f}"])

    # 요약 JSON 저장
    hist_data = json.loads(history_path.read_text(encoding="utf-8"))
    summary = {
        "model": "U-Net (ResNet-34, ImageNet)",
        "loss": "BCE + Per-Image Soft Dice Loss",
        "scheduler": "CosineAnnealingLR",
        "drop_last": True,
        "amp": True,
        "epochs": cfg["train"]["epochs"],
        "best_epoch": hist_data["best_epoch"],
        "best_val_dice": hist_data["best_val_dice"],
        "test_mean_iou": float(np.mean([r["iou"] for r in records])),
        "test_mean_dice": float(np.mean([r["dice"] for r in records])),
        "num_test_images": len(records),
    }
    (out_dir / "test_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # 성공 사례(상위 4장) 및 실패/취약 사례(하위 4장) 시각화
    sorted_recs = sorted(records, key=lambda x: x["dice"], reverse=True)
    plot_case_grid(sorted_recs[:4], "Success", out_dir / "success_cases.png")
    plot_case_grid(list(reversed(sorted_recs[-4:])), "Failure", out_dir / "failure_cases.png")
    save_training_curves(history_path, out_dir / "training_curves.png")
    print("unet_resnet34 평가 산출물 갱신 완료:", summary)


if __name__ == "__main__":
    cfg = load_config()
    evaluate_and_export(cfg, ROOT_DIR / "outputs" / "unet_resnet34")
