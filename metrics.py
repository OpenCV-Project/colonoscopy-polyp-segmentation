"""Image-wise IoU and Dice for binary segmentation masks."""

from statistics import fmean

import numpy as np


def score_mask(prediction, target):
    """Return IoU and Dice for one pair of 2D binary masks."""
    prediction = _as_binary_mask(prediction, "예측 마스크")
    target = _as_binary_mask(target, "정답 마스크")
    if prediction.shape != target.shape:
        raise ValueError(f"마스크 크기가 다릅니다. 예측 {prediction.shape}, 정답 {target.shape}")

    intersection = int(np.count_nonzero(prediction & target))
    foreground_total = int(np.count_nonzero(prediction)) + int(np.count_nonzero(target))
    if foreground_total == 0:
        # 두 마스크가 모두 배경이면 픽셀 단위 정답과 완전히 일치한다.
        return {"iou": 1.0, "dice": 1.0}

    union = foreground_total - intersection
    return {
        "iou": intersection / union,
        "dice": 2 * intersection / foreground_total,
    }


def score_masks(predictions, targets):
    """Return individual scores and their arithmetic mean across images."""
    per_image = [
        score_mask(prediction, target)
        for prediction, target in zip(predictions, targets, strict=True)
    ]
    if not per_image:
        raise ValueError("평가할 예측·정답 마스크 쌍이 없습니다.")

    return {
        "per_image": per_image,
        "mean_iou": fmean(scores["iou"] for scores in per_image),
        "mean_dice": fmean(scores["dice"] for scores in per_image),
    }


def _as_binary_mask(mask, name):
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError(f"{name}는 비어 있지 않은 2차원 배열이어야 합니다.")
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name}는 0(배경)과 1(용종)로 이진화해야 합니다.")
    return mask.astype(bool, copy=False)
