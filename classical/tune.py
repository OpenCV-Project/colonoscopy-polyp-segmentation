"""검증 세트에서 파라미터 조합을 모두 시험해 평균 Dice가 가장 높은 조합을 고른다.

테스트 세트는 읽지 않는다. 고른 조합은 best_params.json에 저장하고, evaluate_test.py는 이 파일만 읽는다.

실행: python -m classical.tune --data-root /path/to/Kvasir-SEG
"""

import csv
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

from classical.config import read_config, save_config
from classical.method import (
    ClassicalParams,
    cluster_pixels,
    extract_color_features,
    refine_mask,
    select_polyp_cluster,
)
from dataset import KvasirDataset
from metrics import score_mask


def build_candidates(config):
    """탐색 범위로 만들 수 있는 모든 파라미터 조합을 만든다. 하한이 상한 이상인 조합은 뺀다."""
    search = config["search"]
    candidates = []
    for n_clusters, min_area_ratio, max_area_ratio, kernel_size in itertools.product(
        search["n_clusters"], search["min_area_ratio"], search["max_area_ratio"], search["kernel_size"]
    ):
        if min_area_ratio >= max_area_ratio:
            continue
        candidates.append(ClassicalParams(
            n_clusters=n_clusters,
            min_area_ratio=min_area_ratio,
            max_area_ratio=max_area_ratio,
            kernel_size=kernel_size,
            seed=config["seed"],
            kmeans_attempts=config["kmeans"]["attempts"],
            kmeans_max_iter=config["kmeans"]["max_iter"],
            kmeans_epsilon=config["kmeans"]["epsilon"],
        ))
    return candidates


def score_candidates(dataset, candidates):
    """조합마다 검증 이미지별 IoU·Dice를 계산해 {조합: [이미지별 점수]}로 반환한다."""
    scores = {candidate: [] for candidate in candidates}
    for image_rgb, target in dataset:
        features = extract_color_features(image_rgb)
        # K-means 결과는 군집 수에만 달라지므로, 군집 수마다 한 번만 군집화해 다른 조합에 재사용한다.
        labels_by_n_clusters = {}
        for candidate in candidates:
            if candidate.n_clusters not in labels_by_n_clusters:
                labels_by_n_clusters[candidate.n_clusters] = cluster_pixels(features, candidate)
            labels = labels_by_n_clusters[candidate.n_clusters]

            mask = select_polyp_cluster(labels, features, candidate)
            prediction = refine_mask(mask, candidate.kernel_size)
            scores[candidate].append(score_mask(prediction, target))
    return scores


def rank_candidates(scores):
    """조합을 평균 Dice가 높은 순으로 정렬해 [(조합, 평균 IoU, 평균 Dice)]로 반환한다."""
    ranked = [
        (candidate, fmean(s["iou"] for s in per_image), fmean(s["dice"] for s in per_image))
        for candidate, per_image in scores.items()
    ]
    # 정렬은 안정적이라 평균 Dice가 같으면 설정 파일의 탐색 순서가 앞선 조합이 먼저 온다.
    ranked.sort(key=lambda row: row[2], reverse=True)
    return ranked


def save_ranking(ranked, path):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow([*asdict(ranked[0][0]), "mean_iou", "mean_dice"])
        for candidate, mean_iou, mean_dice in ranked:
            writer.writerow([*asdict(candidate).values(), mean_iou, mean_dice])


def main():
    config = read_config()
    candidates = build_candidates(config)
    dataset = KvasirDataset(config["data_root"], "val")
    print(f"검증 {len(dataset)}장 × 조합 {len(candidates)}개를 평가합니다.")
    ranked = rank_candidates(score_candidates(dataset, candidates))

    output_dir = Path(config["output_dir"]) / "tune"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir)
    save_ranking(ranked, output_dir / "tuning_results.csv")
    best, mean_iou, mean_dice = ranked[0]
    (output_dir / "best_params.json").write_text(json.dumps(asdict(best), indent=2), encoding="utf-8")
    print(f"최적 조합: {best}\n검증 평균 IoU {mean_iou:.4f}, Dice {mean_dice:.4f}")


if __name__ == "__main__":
    main()
