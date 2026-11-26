"""
저장된 U-Net (ResNet-34) 베스트 체크포인트를 불러와 테스트셋(150장)을 재평가하고,
DeepLabv3+ 산출물 형식에 맞춰 JSON/CSV 요약 파일과 성공/실패 사례 시각화 이미지를 저장하는 스크립트.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

# 저장소 루트 폴더(dataset.py, metrics.py, splits.csv)를 모듈 탐색 경로에 추가
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
  sys.path.append(str(REPO_DIR))

# 같은 unet 폴더의 train_unet.py에서 데이터 래퍼와 모델 생성 함수를 재사용
UNET_DIR = Path(__file__).resolve().parent
if str(UNET_DIR) not in sys.path:
  sys.path.append(str(UNET_DIR))

from dataset import KvasirDataset
from metrics import score_masks
from train_unet import IMAGENET_MEAN, IMAGENET_STD, TorchKvasirDataset, build_unet


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
  """
  정규화된 이미지 텐서 (C, H, W)를 화면 출력 및 오버레이용 RGB uint8 배열 (H, W, 3)로 되돌린다.
  """
  img = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
  img = (img * IMAGENET_STD) + IMAGENET_MEAN
  img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
  return img


def make_overlay(
    rgb_img: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray
) -> np.ndarray:
  """
  정답 마스크(GT=초록)와 모델 예측 마스크(Pred=빨강)를 원본 이미지 위에 반투명 색상으로 겹친다.
  """
  overlay = rgb_img.astype(np.float32).copy()
  alpha = 0.45

  gt_bool = gt_mask > 0
  pred_bool = pred_mask > 0

  green = np.array([0.0, 255.0, 0.0], dtype=np.float32)
  overlay[gt_bool] = (1 - alpha) * overlay[gt_bool] + alpha * green

  red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
  overlay[pred_bool] = (1 - alpha) * overlay[pred_bool] + alpha * red

  return np.clip(overlay, 0, 255).astype(np.uint8)


def save_case_grid(
    cases: list[dict], title: str, save_path: Path, model_label: str = "U-Net Pred"
) -> None:
  """
  성공/실패 사례 여러 장을 [Original | Ground Truth | Pred | Overlay] 4열 그리드 이미지로 저장한다.
  """
  if not cases:
    return

  n_rows = len(cases)
  fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4.2 * n_rows))
  if n_rows == 1:
    axes = np.expand_dims(axes, axis=0)

  col_titles = ["Original", "Ground Truth", model_label, "Overlay (GT=Green, Pred=Red)"]
  for col_idx, col_title in enumerate(col_titles):
    axes[0, col_idx].set_title(col_title, fontsize=12, fontweight="bold", pad=10)

  for row_idx, item in enumerate(cases):
    rgb = item["image"]
    gt = item["gt"]
    pred = item["pred"]
    overlay = make_overlay(rgb, gt, pred)

    row_imgs = [rgb, gt * 255, pred * 255, overlay]
    cmaps = [None, "gray", "gray", None]

    for col_idx, (img_data, cmap) in enumerate(zip(row_imgs, cmaps)):
      ax = axes[row_idx, col_idx]
      ax.imshow(img_data, cmap=cmap)
      ax.set_xticks([])
      ax.set_yticks([])

      if col_idx == 0:
        ylabel_text = (
            f"{item['filename']}\n"
            f"IoU={item['iou']:.3f}\n"
            f"Dice={item['dice']:.3f}"
        )
        ax.set_ylabel(ylabel_text, fontsize=9, fontweight="bold")

  fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
  plt.tight_layout(rect=[0.05, 0.0, 1.0, 0.97])
  plt.savefig(save_path, dpi=150)
  plt.close(fig)


@torch.no_grad()
def run_evaluation(
    dataset_dir: str = "/content/drive/MyDrive/Kvasir-SEG",
    csv_path: str = str(REPO_DIR / "splits.csv"),
    ckpt_path: str = "/content/drive/MyDrive/cv_project/checkpoints/unet_resnet34_best.pth",
    output_dir: str = str(UNET_DIR / "outputs" / "unet_resnet34"),
    batch_size: int = 16,
    threshold: float = 0.5,
    top_k: int = 3,
) -> None:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[평가 시작] 실행 장치: {device}")

  out_dir = Path(output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  test_dataset = TorchKvasirDataset(
      dataset_dir, "test", csv_path=csv_path, augment=False
  )
  raw_test_dataset = KvasirDataset(
      dataset_dir, "test", csv_path=csv_path, augment=False
  )
  test_loader = DataLoader(
      test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
  )

  model = build_unet().to(device)
  ckpt = torch.load(ckpt_path, map_location=device)
  state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
  model.load_state_dict(state_dict)
  model.eval()

  ckpt_epoch = ckpt.get("epoch", None) if isinstance(ckpt, dict) else None
  ckpt_val_dice = ckpt.get("val_dice", None) if isinstance(ckpt, dict) else None
  ckpt_val_iou = ckpt.get("val_iou", None) if isinstance(ckpt, dict) else None
  print(f"체크포인트 로드 완료: {ckpt_path} (epoch={ckpt_epoch}, val_dice={ckpt_val_dice})")

  all_preds = []
  all_gts = []
  per_image_records = []
  vis_records = []

  global_idx = 0
  for batch_images, batch_masks in test_loader:
    batch_images_gpu = batch_images.to(device)
    logits = model(batch_images_gpu)
    probs = torch.sigmoid(logits)
    pred_masks = (probs >= threshold).cpu().numpy().astype(np.uint8)
    gt_masks = batch_masks.numpy().astype(np.uint8)

    for i in range(batch_images.size(0)):
      pred_2d = pred_masks[i, 0]
      gt_2d = gt_masks[i, 0]

      img_scores = score_masks([pred_2d], [gt_2d])
      img_iou = float(img_scores["mean_iou"])
      img_dice = float(img_scores["mean_dice"])

      filename = f"sample_{global_idx:03d}.jpg"
      if hasattr(raw_test_dataset, "df") and "image_name" in raw_test_dataset.df.columns:
        filename = str(raw_test_dataset.df.iloc[global_idx]["image_name"])
      elif hasattr(raw_test_dataset, "rows") and len(raw_test_dataset.rows) > global_idx:
        row = raw_test_dataset.rows[global_idx]
        if isinstance(row, dict):
          filename = str(row.get("image_name") or row.get("filename") or filename)

      all_preds.append(pred_2d)
      all_gts.append(gt_2d)

      per_image_records.append({
          "index": global_idx,
          "filename": filename,
          "iou": round(img_iou, 6),
          "dice": round(img_dice, 6),
          "gt_pixels": int(gt_2d.sum()),
          "pred_pixels": int(pred_2d.sum()),
      })

      vis_records.append({
          "index": global_idx,
          "filename": filename,
          "iou": img_iou,
          "dice": img_dice,
          "image": denormalize_image(batch_images[i]),
          "gt": gt_2d,
          "pred": pred_2d,
      })

      global_idx += 1

  overall_scores = score_masks(all_preds, all_gts)
  summary_data = {
      "model": "U-Net (ResNet-34)",
      "encoder_name": "resnet34",
      "encoder_weights": "imagenet",
      "threshold": threshold,
      "num_test_images": len(all_preds),
      "mean_iou": round(float(overall_scores["mean_iou"]), 6),
      "mean_dice": round(float(overall_scores["mean_dice"]), 6),
      "checkpoint_path": str(ckpt_path),
      "best_epoch": ckpt_epoch,
      "best_val_iou": round(float(ckpt_val_iou), 6) if ckpt_val_iou is not None else None,
      "best_val_dice": round(float(ckpt_val_dice), 6) if ckpt_val_dice is not None else None,
  }

  summary_path = out_dir / "test_summary.json"
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

  csv_out_path = out_dir / "test_per_image_scores.csv"
  with open(csv_out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["index", "filename", "iou", "dice", "gt_pixels", "pred_pixels"],
    )
    writer.writeheader()
    writer.writerows(per_image_records)

  sorted_by_dice = sorted(vis_records, key=lambda x: x["dice"])
  failure_cases = sorted_by_dice[:top_k]
  success_cases = list(reversed(sorted_by_dice[-top_k:]))

  save_case_grid(
      success_cases,
      title=f"U-Net (ResNet-34) - Success Top {len(success_cases)}",
      save_path=out_dir / "success_cases.png",
      model_label="U-Net Pred",
  )
  save_case_grid(
      failure_cases,
      title=f"U-Net (ResNet-34) - Failure Bottom {len(failure_cases)}",
      save_path=out_dir / "failure_cases.png",
      model_label="U-Net Pred",
  )

  print(f"테스트 전체 결과 -> Mean IoU: {summary_data['mean_iou']:.4f} | Mean Dice: {summary_data['mean_dice']:.4f}")
  print(f"저장 완료 폴더: {out_dir}")
  for p in sorted(out_dir.iterdir()):
    print(f"  - {p.name}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="U-Net 테스트셋 평가 및 시각화 생성")
  parser.add_argument("--dataset-dir", default="/content/drive/MyDrive/Kvasir-SEG")
  parser.add_argument("--csv-path", default=str(REPO_DIR / "splits.csv"))
  parser.add_argument(
      "--ckpt-path",
      default="/content/drive/MyDrive/cv_project/checkpoints/unet_resnet34_best.pth",
  )
  parser.add_argument(
      "--output-dir", default=str(UNET_DIR / "outputs" / "unet_resnet34")
  )
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument("--top-k", type=int, default=3)
  args = parser.parse_args()

  run_evaluation(
      dataset_dir=args.dataset_dir,
      csv_path=args.csv_path,
      ckpt_path=args.ckpt_path,
      output_dir=args.output_dir,
      batch_size=args.batch_size,
      threshold=args.threshold,
      top_k=args.top_k,
  )
