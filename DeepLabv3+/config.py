"""DeepLabv3+ 학습 및 평가에 공통으로 사용하는 설정(하이퍼파라미터·경로 유틸) 모듈."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    """실험 조건을 한곳에서 관리하는 설정 클래스.

    U-Net 등 다른 모델과 공정하게 비교하기 위해 인코더, 배치 크기, 에폭,
    학습률, 임계값, 랜덤 시드를 명시적으로 고정합니다.
    """

    # ResNet-34 백본 + ImageNet 사전학습 가중치 사용 (계획서 공통 조건)
    encoder_name: str = "resnet34"
    encoder_weights: str = "imagenet"

    # Colab T4 GPU 메모리와 BatchNorm 안정성을 고려한 배치 크기 (최소 2 이상 필수)
    batch_size: int = 16
    epochs: int = 30

    # AdamW 옵티마이저 학습률 및 가중치 감쇠(L2 정규화)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # 손실 함수 선택: 'bce' (1차 베이스라인) 또는 'bce_dice' (개선 실험)
    loss_name: str = "bce"

    # 시그모이드 확률값을 0/1 이진 마스크로 변환할 기본 임계값 (검증 세트에서 튜닝 가능)
    threshold: float = 0.5

    # 재현성을 위한 고정 난수 시드
    seed: int = 42

    # DataLoader 병렬 로딩 워커 수 (Colab 환경 기본 권장값: 2)
    num_workers: int = 2

    # CUDA GPU 사용 시 혼합 정밀도(AMP, FP16) 학습으로 속도 향상 및 메모리 절약
    use_amp: bool = True

    def to_dict(self) -> dict:
        """설정을 JSON 저장 가능한 딕셔너리로 변환합니다."""
        return asdict(self)


def ensure_dir(path: str | Path) -> Path:
    """출력 디렉터리가 없으면 부모 폴더까지 포함해 생성하고 Path 객체로 반환합니다."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
