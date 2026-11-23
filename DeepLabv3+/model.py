"""DeepLabv3+ 모델 생성, ImageNet 전처리, 손실 함수, 단일 이미지 추론 모듈."""

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# ImageNet 사전학습 가중치(ResNet-34)에 맞춘 채널별(R, G, B) 평균과 표준편차
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 프로젝트 공통 입력 해상도 (가로 256 x 세로 256)
MODEL_INPUT_SIZE = (256, 256)


def select_device() -> torch.device:
    """사용 가능한 연산 장치를 CUDA GPU -> Apple Silicon(MPS) -> CPU 순으로 선택합니다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_deeplabv3plus(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """이진 분할(배경 vs 용종)을 위한 DeepLabv3+ 모델을 생성합니다.

    - encoder_weights='imagenet': ImageNet 사전학습 가중치로 초기화하며,
      별도 동결(freeze) 없이 인코더+디코더 전체를 Kvasir-SEG로 파인튜닝합니다.
    - in_channels=3: RGB 3채널 입력
    - classes=1: 픽셀별 용종일 확률(logit) 1채널 출력
    """
    return smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """비교표 작성에 필요한 전체 파라미터 수와 학습 가능 파라미터 수를 계산합니다."""
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def normalize_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    """RGB uint8 (H, W, 3) 이미지를 ImageNet 통계로 정규화하고 (3, H, W)로 변환합니다."""
    # 0~255 정수 픽셀값을 0.0~1.0 실수로 스케일링
    normalized = image_rgb.astype(np.float32) / 255.0
    # 채널별 평균을 빼고 표준편차로 나누어 사전학습 인코더 분포에 맞춤
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD
    # PyTorch 텐서 입력 형식인 Channel-First (C, H, W)로 축 순서 변경
    return np.transpose(normalized, (2, 0, 1))


def preprocess_raw_bgr(image_bgr: np.ndarray) -> torch.Tensor:
    """원본 BGR 이미지를 리사이즈·RGB 변환·정규화해 모델 입력 텐서 (1, 3, 256, 256)로 만듭니다.

    main/benchmark.py로 이미지당 처리 시간(ms)을 측정할 때 전처리 시간까지 포함하기 위해 사용합니다.
    """
    resized = cv2.resize(image_bgr, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(normalize_rgb_image(rgb))
    # 배치 차원(1) 추가 -> (1, 3, 256, 256)
    return tensor.unsqueeze(0)


@torch.inference_mode()
def predict_prob_map(
    image_rgb_256: np.ndarray,
    model: nn.Module,
    device: torch.device,
) -> np.ndarray:
    """이미 256x256으로 리사이즈된 RGB 이미지에서 용종 확률 맵 (H, W) [0.0~1.0]을 예측합니다."""
    tensor = torch.from_numpy(normalize_rgb_image(image_rgb_256)).unsqueeze(0).to(device)
    logits = model(tensor)
    # 모델 출력(logit)에 시그모이드를 적용해 확률(0~1)로 변환
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0)
    return probs.detach().cpu().numpy()


@torch.inference_mode()
def predict_mask(
    image_rgb_256: np.ndarray,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """256x256 RGB 이미지에서 0(배경)과 1(용종)으로 이루어진 uint8 이진 마스크 (H, W)를 반환합니다."""
    probs = predict_prob_map(image_rgb_256, model, device)
    return (probs >= threshold).astype(np.uint8)


@torch.inference_mode()
def predict_mask_from_raw(
    raw_bgr_image: np.ndarray,
    model: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """디코딩된 원본 BGR 이미지 1장을 받아 전처리 + 추론 + 이진화를 한 번에 수행합니다.

    main/benchmark.py의 benchmark_image_time()에 전달하는 콜백 함수로 사용됩니다.
    """
    tensor = preprocess_raw_bgr(raw_bgr_image).to(device)
    logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0)
    return (probs >= threshold).to(torch.uint8).cpu().numpy()


class BCEDiceLoss(nn.Module):
    """픽셀 단위 이진 크로스 엔트로피(BCE)와 영역 겹침(Soft Dice)을 결합한 손실 함수.

    용종이 작은 영상에서는 배경 픽셀이 압도적으로 많아 BCE만 쓰면 과소 분할되기 쉬우므로,
    Dice 점수를 직접 최대화하는 항(1 - Dice)을 함께 최적화합니다.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth  # 0으로 나누기 방지용 작은 상수
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        # 미분 가능한 Soft Dice 계산을 위해 시그모이드 확률값 사용
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)  # 배치(0)별 각 이미지의 (C, H, W) 전체에 대해 합산
        intersection = torch.sum(probs * targets, dim=dims)
        union_sum = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (union_sum + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def build_loss(loss_name: str) -> nn.Module:
    """설정 이름('bce' 또는 'bce_dice')에 맞는 손실 함수 인스턴스를 반환합니다."""
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "bce_dice":
        return BCEDiceLoss()
    raise ValueError(f"지원하지 않는 손실 함수입니다: {loss_name}")
