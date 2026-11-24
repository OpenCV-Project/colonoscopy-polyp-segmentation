"""DeepLabv3+ 학습 루프, 체크포인트 저장, 검증 세트 기반 임계값 튜닝 모듈."""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import TrainConfig, ensure_dir
from model import build_deeplabv3plus, build_loss, normalize_rgb_image, select_device


def _import_common_modules(main_dir: str | Path):
    """팀 공통 코드(main/dataset.py, main/metrics.py)를 동적으로 불러옵니다."""
    main_path = str(Path(main_dir).resolve())
    if main_path not in sys.path:
        sys.path.insert(0, main_path)
    from dataset import KvasirDataset
    from metrics import score_masks

    return KvasirDataset, score_masks


def set_seed(seed: int) -> None:
    """Python, NumPy, PyTorch(CPU/GPU)의 난수 시드를 고정해 실험 재현성을 확보합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class KvasirTorchDataset(Dataset):
    """공통 KvasirDataset(NumPy uint8 반환)을 PyTorch 학습용 텐서로 변환하는 어댑터."""

    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # KvasirDataset은 (256, 256, 3) RGB uint8 이미지와 (256, 256) 0/1 uint8 마스크를 반환
        image_rgb, mask = self.base_dataset[index]
        image_tensor = torch.from_numpy(normalize_rgb_image(image_rgb))
        # 손실 함수 입력 규격 (1, H, W) float32에 맞추기 위해 채널 차원 추가
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        return image_tensor, mask_tensor


def build_dataloaders(
    dataset_dir: str | Path,
    main_dir: str | Path,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """고정 분할(main/splits.csv)을 사용해 train/val/test DataLoader를 생성합니다."""
    KvasirDataset, _ = _import_common_modules(main_dir)
    csv_path = Path(main_dir) / "splits.csv"

    # 학습 세트에만 좌우 반전 및 ±10도 회전 증강 적용 (검증/테스트 세트는 원본 유지)
    train_base = KvasirDataset(dataset_dir, "train", csv_path=csv_path, augment=True)
    val_base = KvasirDataset(dataset_dir, "val", csv_path=csv_path, augment=False)
    test_base = KvasirDataset(dataset_dir, "test", csv_path=csv_path, augment=False)

    pin_memory = torch.cuda.is_available()

    # DeepLabv3+의 ASPP 모듈이 BatchNorm2d를 사용하므로 학습 시 마지막 배치가 1장이 되면 오류가 발생함.
    # 이를 방지하기 위해 train_loader에는 반드시 drop_last=True를 설정합니다.
    train_loader = DataLoader(
        KvasirTorchDataset(train_base),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        KvasirTorchDataset(val_base),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        KvasirTorchDataset(test_base),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    """한 에폭 동안 모델을 학습하고 샘플 수 가중 평균 학습 손실을 반환합니다."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    use_amp = scaler is not None and device.type == "cuda"

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # CUDA 환경에서는 자동 혼합 정밀도(AMP)로 연산 속도를 높임
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    return total_loss / max(num_samples, 1)


@torch.inference_mode()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    score_masks_fn,
    threshold: float = 0.5,
) -> dict[str, float]:
    """검증 세트에 대해 손실과 공통 지표(mean IoU, mean Dice)를 계산합니다."""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    predictions = []
    targets = []

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, masks)

        # 시그모이드 확률값을 threshold 기준으로 0/1 uint8 마스크로 이진화
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        pred_batch = (probs >= threshold).astype(np.uint8)
        target_batch = masks.squeeze(1).cpu().numpy().astype(np.uint8)

        predictions.extend(pred_batch)
        targets.extend(target_batch)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        num_samples += batch_size

    # 팀 공통 평가 함수(main/metrics.py)로 이미지별 IoU/Dice 산술 평균 계산
    scores = score_masks_fn(predictions, targets)
    return {
        "loss": total_loss / max(num_samples, 1),
        "mean_iou": scores["mean_iou"],
        "mean_dice": scores["mean_dice"],
    }


def train_model(
    config: TrainConfig,
    dataset_dir: str | Path,
    main_dir: str | Path,
    output_dir: str | Path,
) -> dict:
    """전체 에폭 학습을 수행하고 검증 평균 Dice 최고 모델 체크포인트를 저장합니다."""
    set_seed(config.seed)
    device = select_device()
    out_path = ensure_dir(output_dir)
    _, score_masks_fn = _import_common_modules(main_dir)

    train_loader, val_loader, _ = build_dataloaders(dataset_dir, main_dir, config)
    model = build_deeplabv3plus(config.encoder_name, config.encoder_weights).to(device)
    criterion = build_loss(config.loss_name)
    # 인코더·디코더 전체 파라미터를 동일한 학습률로 함께 파인튜닝
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # CosineAnnealingLR로 학습률을 에폭에 따라 부드럽게 감소
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler("cuda") if (config.use_amp and device.type == "cuda") else None

    history = []
    best_dice = -1.0
    best_epoch = 0
    ckpt_path = out_path / "best_model.pth"

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, score_masks_fn, threshold=config.threshold
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["mean_iou"],
            "val_dice": val_metrics["mean_dice"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        # 검증 평균 Dice가 갱신될 때만 최고 성능 가중치를 저장 (테스트 세트는 절대 보지 않음)
        if val_metrics["mean_dice"] > best_dice:
            best_dice = val_metrics["mean_dice"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": best_dice,
                    "val_iou": val_metrics["mean_iou"],
                    "config": config.to_dict(),
                },
                ckpt_path,
            )

        print(
            f"[Epoch {epoch:02d}/{config.epochs:02d}] "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_IoU={val_metrics['mean_iou']:.4f} | val_Dice={val_metrics['mean_dice']:.4f}"
        )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "checkpoint_path": str(ckpt_path),
        "config": config.to_dict(),
        "history": history,
    }
    (out_path / "train_history.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


@torch.inference_mode()
def find_best_threshold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    main_dir: str | Path,
    device: torch.device,
    candidates: tuple[float, ...] = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70),
) -> dict:
    """검증 세트에서만 최적의 이진화 임계값을 탐색합니다 (테스트 세트 누수 방지)."""
    _, score_masks_fn = _import_common_modules(main_dir)
    model.eval()
    prob_maps = []
    targets = []

    # 검증 세트 전체에 대해 확률 맵을 한 번만 추론해 메모리에 모아둠
    for images, masks in val_loader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        prob_maps.extend(probs)
        targets.extend(masks.squeeze(1).numpy().astype(np.uint8))

    # 후보 임계값별로 검증 평균 Dice를 비교해 최고 임계값 선택
    results = []
    best_record = None
    for thresh in candidates:
        preds = [(prob >= thresh).astype(np.uint8) for prob in prob_maps]
        scores = score_masks_fn(preds, targets)
        record = {
            "threshold": thresh,
            "val_iou": scores["mean_iou"],
            "val_dice": scores["mean_dice"],
        }
        results.append(record)
        if best_record is None or record["val_dice"] > best_record["val_dice"]:
            best_record = record

    return {"best": best_record, "all_candidates": results}
