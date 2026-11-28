"""학습 세트에서 '용종이 배경보다 붉은가', '용종 면적은 몇 %인가'를 확인한다.

고전 기법은 학습하지 않으므로 학습 세트는 설계 근거를 확인하는 데만 쓴다. 파라미터는 tune.py가 검증 세트로 고른다.

실행: python -m classical.explore_train --data-root /path/to/Kvasir-SEG
"""

import json
from pathlib import Path

import cv2 as cv
import numpy as np

from classical.config import read_config, save_config
from dataset import KvasirDataset


# 이미지 가장자리의 검은 테두리는 장벽이 아니므로 배경에서 뺀다.
DARK_BORDER_MAX_VALUE = 30


def main():
    config = read_config()
    polyp_higher = {"a_star": [], "b_star": [], "saturation": [], "value": []}
    area_ratios = []
    for image_rgb, mask in KvasirDataset(config["data_root"], "train"):
        lab = cv.cvtColor(image_rgb, cv.COLOR_RGB2LAB)
        hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
        channels = (lab[..., 1], lab[..., 2], hsv[..., 1], hsv[..., 2])

        is_polyp = mask == 1
        is_background = ~is_polyp & (hsv[..., 2] >= DARK_BORDER_MAX_VALUE)
        for name, channel in zip(polyp_higher, channels):
            polyp_higher[name].append(channel[is_polyp].mean() > channel[is_background].mean())
        area_ratios.append(is_polyp.mean())

    summary = {
        "polyp_higher_ratio": {name: float(np.mean(values)) for name, values in polyp_higher.items()},
        "polyp_area_ratio_percentiles": {
            str(p): float(np.percentile(area_ratios, p)) for p in (5, 25, 50, 75, 95, 100)
        },
    }
    output_dir = Path(config["output_dir"]) / "explore_train"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
