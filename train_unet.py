"""
Kvasir-SEG 데이터셋(700/150/150 고정 분할)을 사용해
ResNet-34 백본 + ImageNet 사전학습 가중치 기반 U-Net을 학습하고 평가하는 스크립트.
"""

from pathlib import Path
import random
import sys

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
  sys.path.append(str(REPO_DIR))

from dataset import KvasirDataset
from metrics import score_masks

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_seed(seed=42):
  """실험 재현성을 위해 난수 시드를 고정한다."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)


class TorchKvasirDataset(Dataset):
  """KvasirDataset(NumPy 배열)을 PyTorch Tensor로 변환하는 래퍼 클래스."""

  def __init__(self, dataset_dir, split, csv_path=None, augment=False):
    kwargs = {"augment": augment}
    if csv_path is not None:
      kwargs["csv_path"] = csv_path
    self.base = KvasirDataset(dataset_dir, split, **kwargs)

  def __len__(self):
    return len(self.base)

  def __getitem__(self, index):
    image, mask = self.base[index]
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous()
    mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
    return image_tensor, mask_tensor


class BCEDiceLoss(nn.Module):
  """BCE 손실과 Dice 손실을 함께 사용하는 이진 분할 손실 함수."""

  def __init__(self, bce_weight=0.5, dice_weight=0.5):
    super().__init__()
    self.bce = nn.BCEWithLogitsLoss()
    self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    self.bce_weight = bce_weight
    self.dice_weight = dice_weight

  def forward(self, logits, targets):
    return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
        logits, targets
    )


def build_unet():
  """ResNet-34 인코더 + ImageNet 가중치 기반 U-Net 모델 생성."""
  return smp.Unet(
      encoder_name="resnet34",
      encoder_weights="imagenet",
      in_channels=3,
      classes=1,
  )


def count_parameters(model):
  """전체 파라미터 수와 학습 가능 파라미터 수 반환."""
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
  """검증/테스트 세트에서 IoU와 Dice 평균 계산."""
  model.eval()
  predictions = []
  targets = []

  for images, masks in loader:
    images = images.to(device)
    logits = model(images)
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).cpu().numpy().astype(np.uint8)
    gts = masks.numpy().astype(np.uint8)

    for pred, gt in zip(preds, gts):
      predictions.append(pred[0])
      targets.append(gt[0])

  return score_masks(predictions, targets)


def run_training(
    dataset_dir="/content/drive/MyDrive/Kvasir-SEG",
    csv_path=str(REPO_DIR / "splits.csv"),
    save_dir="/content/drive/MyDrive/cv_project/checkpoints",
    batch_size=16,
    epochs=30,
    lr=1e-4,
    threshold=0.5,
    seed=42,
):
  set_seed(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"실행 장치: {device}")

  train_dataset = TorchKvasirDataset(
      dataset_dir, "train", csv_path=csv_path, augment=True
  )
  val_dataset = TorchKvasirDataset(
      dataset_dir, "val", csv_path=csv_path, augment=False
  )
  test_dataset = TorchKvasirDataset(
      dataset_dir, "test", csv_path=csv_path, augment=False
  )

  train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      pin_memory=True,
  )
  val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      pin_memory=True,
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      pin_memory=True,
  )

  model = build_unet().to(device)
  total_params, trainable_params = count_parameters(model)
  print(
      f"U-Net (ResNet-34) 전체 파라미터 수: {total_params:,} (학습 가능:"
      f" {trainable_params:,})"
  )

  criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

  save_path = Path(save_dir)
  save_path.mkdir(parents=True, exist_ok=True)
  best_ckpt_path = save_path / "unet_resnet34_best.pth"

  best_val_dice = -1.0
  best_epoch = 0

  for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
      images = images.to(device)
      masks = masks.to(device)

      optimizer.zero_grad()
      logits = model(images)
      loss = criterion(logits, masks)
      loss.backward()
      optimizer.step()

      running_loss += loss.item() * images.size(0)

    train_loss = running_loss / len(train_dataset)
    val_result = evaluate(model, val_loader, device, threshold=threshold)
    val_iou = val_result["mean_iou"]
    val_dice = val_result["mean_dice"]

    print(
        f"[Epoch {epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val"
        f" IoU: {val_iou:.4f} | Val Dice: {val_dice:.4f}"
    )

    if val_dice > best_val_dice:
      best_val_dice = val_dice
      best_epoch = epoch
      torch.save(
          {
              "epoch": epoch,
              "model_state_dict": model.state_dict(),
              "val_iou": val_iou,
              "val_dice": val_dice,
              "threshold": threshold,
          },
          best_ckpt_path,
      )
      print(
          f"  --> 최고 검증 Dice 갱신 ({best_val_dice:.4f}), 체크포인트 저장:"
          f" {best_ckpt_path}"
      )

  print(
      f"\n학습 완료! 최고 검증 성능: Epoch {best_epoch} (Val Dice:"
      f" {best_val_dice:.4f})"
  )

  checkpoint = torch.load(best_ckpt_path, map_location=device)
  model.load_state_dict(checkpoint["model_state_dict"])
  test_result = evaluate(model, test_loader, device, threshold=threshold)
  print(
      f"[Test Set (150장)] Mean IoU: {test_result['mean_iou']:.4f} | Mean"
      f" Dice: {test_result['mean_dice']:.4f}"
  )


if __name__ == "__main__":
  run_training()
