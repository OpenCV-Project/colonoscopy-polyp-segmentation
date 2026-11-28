"""검증 세트에서 고른 파라미터로 테스트 세트를 한 번 평가하고 이미지당 처리 시간을 잰다.

파라미터는 tune.py가 저장한 best_params.json에서만 읽는다. 테스트 결과를 보고 파라미터를 바꾸지 않는다.

실행: python -m classical.evaluate_test --data-root /path/to/Kvasir-SEG
"""

import csv
import json
import platform
from pathlib import Path

import cv2 as cv

from benchmark import benchmark_image_time
from classical.config import read_config, save_config
from classical.method import ClassicalParams, predict_mask, predict_mask_from_raw
from dataset import KvasirDataset
from metrics import score_masks


TIMING_SCOPE = "포함: 256×256 리사이즈·RGB 변환부터 모폴로지까지. 제외: 디스크 읽기, 지표 계산."


def decode_raw_images(samples):
    """시간 측정에서 디스크 읽기를 빼기 위해 원본 BGR 이미지를 미리 읽어 둔다."""
    return [cv.imread(str(image_path), cv.IMREAD_COLOR) for image_path, _ in samples]


def save_per_image_scores(samples, per_image, path):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "iou", "dice"])
        for (image_path, _), score in zip(samples, per_image, strict=True):
            writer.writerow([image_path.name, score["iou"], score["dice"]])


def main():
    config = read_config()
    output_root = Path(config["output_dir"])
    params = ClassicalParams(**json.loads((output_root / "tune" / "best_params.json").read_text()))
    dataset = KvasirDataset(config["data_root"], "test")

    predictions = []
    targets = []
    for image_rgb, target in dataset:
        predictions.append(predict_mask(image_rgb, params))
        targets.append(target)
    scores = score_masks(predictions, targets)

    timing = benchmark_image_time(
        lambda raw_bgr_image: predict_mask_from_raw(raw_bgr_image, params),
        decode_raw_images(dataset.samples),
        device="cpu",
        warmup=config["benchmark_warmup"],
    )

    summary = {
        "num_images": len(dataset),
        "mean_iou": scores["mean_iou"],
        "mean_dice": scores["mean_dice"],
        "mean_ms_per_image": timing["mean_ms_per_image"],
        "device": f"cpu ({platform.platform()}, OpenCV 스레드 {cv.getNumThreads()}개)",
        "timing_scope": TIMING_SCOPE,
    }

    output_dir = output_root / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    save_per_image_scores(dataset.samples, scores["per_image"], output_dir / "test_per_image_scores.csv")
    (output_dir / "test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
