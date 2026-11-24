"""테스트 세트 정량 평가, 처리 시간 벤치마크, 정성 시각화, CVC-ClinicDB 외부 평가 모듈."""

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import ensure_dir
from model import (
    MODEL_INPUT_SIZE,
    count_parameters,
    predict_mask,
    predict_mask_from_raw,
)


def _import_common_modules(main_dir: str | Path):
    """팀 공통 데이터 로더, 평가 지표, 벤치마크 모듈을 불러옵니다."""
    main_path = str(Path(main_dir).resolve())
    if main_path not in sys.path:
        sys.path.insert(0, main_path)
    from benchmark import benchmark_image_time
    from dataset import KvasirDataset
    from metrics import score_masks

    return KvasirDataset, score_masks, benchmark_image_time


def _decode_raw_bgr_images(samples: list[tuple[Path, Path]]) -> list[np.ndarray]:
    """디스크 I/O 시간을 제외하고 순수 추론 시간만 측정하기 위해 원본 이미지를 미리 디코딩합니다."""
    images = []
    for image_path, _ in samples:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"이미지를 읽을 수 없습니다: {image_path}")
        images.append(image)
    return images


def evaluate_kvasir_test(
    model: torch.nn.Module,
    dataset_dir: str | Path,
    main_dir: str | Path,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: str | Path | None = None,
    warmup: int = 5,
) -> dict:
    """Kvasir-SEG 테스트 세트(150장)에 대한 IoU, Dice, 파라미터 수, 처리 시간(ms)을 평가합니다."""
    KvasirDataset, score_masks, benchmark_image_time = _import_common_modules(main_dir)
    csv_path = Path(main_dir) / "splits.csv"
    test_dataset = KvasirDataset(dataset_dir, "test", csv_path=csv_path, augment=False)

    # 1. 테스트 세트 전체 예측 수행 (공통 256x256 입력 기준)
    model.eval()
    images_rgb = []
    targets = []
    predictions = []
    for index in range(len(test_dataset)):
        image_rgb, mask = test_dataset[index]
        pred_mask = predict_mask(image_rgb, model, device, threshold=threshold)
        images_rgb.append(image_rgb)
        targets.append(mask)
        predictions.append(pred_mask)

    # 2. 이미지별 IoU/Dice 및 산술 평균 계산
    score_result = score_masks(predictions, targets)

    # 3. 동일 조건(원본 디코딩 이미지 1장씩 전처리+추론+이진화)으로 처리 시간(ms/img) 측정
    raw_bgr_images = _decode_raw_bgr_images(test_dataset.samples)
    bench_device = "cuda" if device.type == "cuda" else "cpu"
    bench_result = benchmark_image_time(
        lambda img: predict_mask_from_raw(img, model, device, threshold=threshold),
        raw_bgr_images,
        device=bench_device,
        warmup=warmup,
    )
    params = count_parameters(model)

    per_image_records = [
        {
            "filename": test_dataset.samples[idx][0].name,
            "iou": score["iou"],
            "dice": score["dice"],
        }
        for idx, score in enumerate(score_result["per_image"])
    ]

    summary = {
        "dataset": "Kvasir-SEG (test)",
        "num_images": len(test_dataset),
        "threshold": threshold,
        "mean_iou": score_result["mean_iou"],
        "mean_dice": score_result["mean_dice"],
        "mean_ms_per_image": bench_result["mean_ms_per_image"],
        "benchmark_device": bench_result["device"],
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "per_image": per_image_records,
    }

    # 결과 디렉터리가 지정되면 이미지별 CSV와 요약 JSON 저장
    if output_dir is not None:
        out_path = ensure_dir(output_dir)
        _save_per_image_csv(per_image_records, out_path / "test_per_image_scores.csv")
        summary_to_save = {k: v for k, v in summary.items() if k != "per_image"}
        (out_path / "test_summary.json").write_text(
            json.dumps(summary_to_save, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return {
        "summary": summary,
        "images_rgb": images_rgb,
        "targets": targets,
        "predictions": predictions,
    }


def _save_per_image_csv(records: list[dict], csv_path: Path) -> None:
    """테스트 이미지별 IoU, Dice 점수를 CSV 파일로 저장합니다."""
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filename", "iou", "dice"])
        writer.writeheader()
        writer.writerows(records)


def select_success_and_failure_indices(
    per_image_records: list[dict],
    top_k: int = 3,
) -> tuple[list[int], list[int]]:
    """Dice 점수 기준으로 상위 top_k(성공 사례)와 하위 top_k(실패 사례) 인덱스를 추출합니다."""
    ranked = sorted(range(len(per_image_records)), key=lambda i: per_image_records[i]["dice"])
    failure_indices = ranked[:top_k]
    success_indices = list(reversed(ranked[-top_k:]))
    return success_indices, failure_indices


def save_qualitative_figure(
    indices: list[int],
    images_rgb: list[np.ndarray],
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    per_image_records: list[dict],
    title: str,
    save_path: str | Path,
) -> None:
    """원본, 정답 마스크, 예측 마스크, 오버레이(GT=초록, 예측=빨강)를 나란히 시각화해 저장합니다."""
    num_rows = len(indices)
    fig, axes = plt.subplots(num_rows, 4, figsize=(14, 3.4 * num_rows))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["Original", "Ground Truth", "DeepLabv3+ Pred", "Overlay (GT=Green, Pred=Red)"]
    for col, col_title in enumerate(col_titles):
        axes[0, col].set_title(col_title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(indices):
        img = images_rgb[idx]
        gt = targets[idx]
        pred = predictions[idx]
        rec = per_image_records[idx]

        # 정답(초록)과 예측(빨강) 영역을 반투명하게 겹쳐 오분할 위치를 한눈에 확인
        overlay = img.copy()
        overlay[gt == 1] = (overlay[gt == 1] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        overlay[pred == 1] = (overlay[pred == 1] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(
            f"{rec['filename']}\nIoU={rec['iou']:.3f}\nDice={rec['dice']:.3f}",
            fontsize=9,
        )
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 2].imshow(pred, cmap="gray")
        axes[row, 3].imshow(overlay)

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _find_cvc_pairs(cvc_dir: Path) -> list[tuple[Path, Path]]:
    """CVC-ClinicDB의 두 가지 흔한 폴더 구조(Original/Ground Truth 또는 images/masks)에서 파일 쌍을 찾습니다."""
    candidate_pairs = [
        (cvc_dir / "Original", cvc_dir / "Ground Truth"),
        (cvc_dir / "images", cvc_dir / "masks"),
    ]
    img_dir, mask_dir = None, None
    for cand_img, cand_mask in candidate_pairs:
        if cand_img.is_dir() and cand_mask.is_dir():
            img_dir, mask_dir = cand_img, cand_mask
            break

    if img_dir is None or mask_dir is None:
        raise FileNotFoundError(
            f"CVC-ClinicDB 폴더 구조를 찾을 수 없습니다: {cvc_dir} "
            "(Original/Ground Truth 또는 images/masks 필요)"
        )

    valid_exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    # 파일명(stem) 기준으로 이미지와 마스크 매칭
    mask_by_stem = {
        p.stem: p for p in mask_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts
    }
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in valid_exts:
            mask_path = mask_by_stem.get(img_path.stem)
            if mask_path is not None:
                pairs.append((img_path, mask_path))

    if not pairs:
        raise ValueError(f"CVC-ClinicDB에서 일치하는 이미지-마스크 쌍을 찾지 못했습니다: {cvc_dir}")
    return pairs


def evaluate_cvc_clinicdb(
    model: torch.nn.Module,
    cvc_dir: str | Path,
    main_dir: str | Path,
    device: torch.device,
    threshold: float = 0.5,
    output_dir: str | Path | None = None,
) -> dict:
    """확장 실험: Kvasir-SEG로 학습한 모델을 CVC-ClinicDB 전체에 추가 학습 없이 적용해 일반화 성능을 평가합니다."""
    _, score_masks, _ = _import_common_modules(main_dir)
    pairs = _find_cvc_pairs(Path(cvc_dir))

    model.eval()
    predictions = []
    targets = []
    for img_path, mask_path in pairs:
        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image_bgr is None or mask_gray is None:
            raise OSError(f"CVC-ClinicDB 파일을 읽을 수 없습니다: {img_path.name}")

        # KvasirDataset과 동일한 규격(256x256, INTER_AREA / INTER_NEAREST)으로 전처리
        image_rgb = cv2.cvtColor(
            cv2.resize(image_bgr, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB,
        )
        mask_bin = (mask_gray >= 128).astype(np.uint8)
        mask_resized = cv2.resize(mask_bin, MODEL_INPUT_SIZE, interpolation=cv2.INTER_NEAREST)

        pred = predict_mask(image_rgb, model, device, threshold=threshold)
        predictions.append(pred)
        targets.append(mask_resized)

    scores = score_masks(predictions, targets)
    summary = {
        "dataset": "CVC-ClinicDB (external)",
        "num_images": len(pairs),
        "threshold": threshold,
        "mean_iou": scores["mean_iou"],
        "mean_dice": scores["mean_dice"],
    }

    if output_dir is not None:
        out_path = ensure_dir(output_dir)
        (out_path / "cvc_clinicdb_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return summary
