"""Lab·HSV 색 특징 + K-means 군집 + 모폴로지로 용종 마스크를 만드는 고전 기법.

학습 없이 이미지 한 장마다 아래 4단계를 거친다.
1. 색 특징: 픽셀마다 Lab의 a*·b*, HSV의 S·V를 영상 안에서 0~1로 정규화한다.
2. 군집화: 색 특징이 비슷한 픽셀끼리 K-means로 묶는다.
3. 군집 선택: 면적 비율이 범위 안에 있는 군집 중 평균 a*(붉은 정도)가 가장 높은 군집을 용종으로 본다.
   개선 실험에서는 a*에 V(밝기)를 value_weight만큼 더한 점수로 고른다. value_weight=0이면 원래 방법과 같다.
4. 정제: 열림으로 작은 잡음을 지우고, 닫힘으로 작은 구멍과 틈을 메운다.
"""

from dataclasses import dataclass

import cv2 as cv
import numpy as np

from dataset import IMAGE_SIZE


# extract_color_features가 쌓는 특징 순서: a*, b*, S, V
A_STAR_INDEX = 0
VALUE_INDEX = 3


@dataclass(frozen=True)
class ClassicalParams:
    """고전 기법 파라미터.

    n_clusters, min_area_ratio, max_area_ratio, kernel_size, value_weight는 검증 세트에서 고른다.
    seed와 kmeans_* 값은 모든 조합에서 같게 고정한다.
    """

    n_clusters: int
    min_area_ratio: float
    max_area_ratio: float
    kernel_size: int
    seed: int
    kmeans_attempts: int
    kmeans_max_iter: int
    kmeans_epsilon: float
    # 개선 실험: 군집을 고를 때 밝기(V)를 더하는 비중. 0이면 원래 방법(a*만 사용)이다.
    value_weight: float = 0.0


def predict_mask(image_rgb, params):
    """256×256 RGB uint8 이미지에서 0(배경)·1(용종) uint8 마스크 (H, W)를 반환한다."""
    features = extract_color_features(image_rgb)
    labels = cluster_pixels(features, params)
    mask = select_polyp_cluster(labels, features, params)
    return refine_mask(mask, params.kernel_size)


def predict_mask_from_raw(raw_bgr_image, params):
    """디코딩한 원본 BGR 이미지를 공통 로더(dataset.py)와 같게 전처리한 뒤 마스크를 예측한다.

    benchmark.py의 benchmark_image_time()에 넘겨 전처리까지 포함한 처리 시간을 잴 때 쓴다.
    """
    resized = cv.resize(raw_bgr_image, IMAGE_SIZE, interpolation=cv.INTER_AREA)
    image_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB)
    return predict_mask(image_rgb, params)


def extract_color_features(image_rgb):
    """픽셀마다 [a*, b*, S, V]를 영상 안에서 0~1로 정규화한 (H, W, 4) float32 배열을 만든다."""
    lab = cv.cvtColor(image_rgb, cv.COLOR_RGB2LAB)
    hsv = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    channels = (lab[..., 1], lab[..., 2], hsv[..., 1], hsv[..., 2])
    # 채널마다 값의 범위가 달라서, 한 채널이 거리 계산을 독차지하지 않도록 0~1로 맞춘다.
    normalized = [
        cv.normalize(channel, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
        for channel in channels
    ]
    return np.stack(normalized, axis=-1)


def cluster_pixels(features, params):
    """색 특징이 비슷한 픽셀끼리 K-means로 묶어 픽셀별 군집 번호 (H, W)를 반환한다."""
    height, width, num_features = features.shape
    samples = features.reshape(-1, num_features)
    criteria = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
        params.kmeans_max_iter,
        params.kmeans_epsilon,
    )
    # K-means는 초기 중심을 무작위로 고르므로, 시드를 고정해 실행할 때마다 같은 결과를 낸다.
    cv.setRNGSeed(params.seed)
    _, labels, _ = cv.kmeans(
        samples, params.n_clusters, None, criteria, params.kmeans_attempts, cv.KMEANS_PP_CENTERS
    )
    return labels.reshape(height, width)


def select_polyp_cluster(labels, features, params):
    """면적 비율이 범위 안에 있는 군집 중 평균 점수가 가장 높은 군집을 0/1 uint8 마스크로 반환한다.

    점수는 a* + value_weight × V이다(value_weight=0이면 a*만 본다).
    조건을 만족하는 군집이 없으면 빈 마스크를 반환한다.
    """
    score = features[..., A_STAR_INDEX] + params.value_weight * features[..., VALUE_INDEX]
    polyp_cluster = None
    highest_mean_score = -np.inf
    for cluster in range(params.n_clusters):
        in_cluster = labels == cluster
        area_ratio = np.count_nonzero(in_cluster) / labels.size
        is_valid_area = params.min_area_ratio <= area_ratio <= params.max_area_ratio
        if not is_valid_area:
            continue

        mean_score = score[in_cluster].mean()
        if mean_score > highest_mean_score:
            polyp_cluster = cluster
            highest_mean_score = mean_score

    if polyp_cluster is None:
        return np.zeros(labels.shape, dtype=np.uint8)
    return (labels == polyp_cluster).astype(np.uint8)


def refine_mask(mask, kernel_size):
    """열림으로 작은 잡음을 지우고, 닫힘으로 작은 구멍과 틈을 메운다. kernel_size=1이면 그대로다."""
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    return cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
