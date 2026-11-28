# U-Net (ResNet-34) 최종 보고서 반영용 정리 (담당: 강승태)

> 본 문서는 `feature/unet` 브랜치의 실험 설정, 정량 평가 수치, 시각화 산출물 위치를 팀 최종 보고서 양식에 맞춰 정리한 초안입니다.

---

## 2.2 U-Net (담당: 강승태)
- **모델**: `segmentation_models_pytorch.Unet` (`encoder_name="resnet34"`, `encoder_weights="imagenet"`)
- **학습 조건**:
  - 입력 해상도 및 분할: `256×256` 리사이즈, ImageNet 정규화, `splits.csv` 동일 분할 (`Train 700 / Val 150 / Test 150`)
  - 손실 함수: `BCE + Per-Image Soft Dice Loss`
  - 옵티마이저 / 스케줄러: `AdamW` (`lr=1e-4`, `weight_decay=1e-4`), `CosineAnnealingLR` (`epochs=30`, `batch_size=16`, `drop_last=True`)
  - 연산 최적화 및 후처리: CUDA `AMP` 적용, 이진화 임계값 `0.5`
- **검증 최고 성능**: Best Val Dice **0.8809** (Epoch 16)
- **Kvasir-SEG 테스트 성능**: Test Mean IoU **0.8309**, Test Mean Dice **0.8937**

---

## 최종 성능 비교표 반영 행

| 방법 | 실행 장치 | 전·후처리 범위 | 파라미터 수 | Mean IoU | Mean Dice | 처리 시간 (ms/img) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| U-Net (ResNet-34) | Colab CUDA GPU | 리사이즈, ImageNet 정규화, 이진화(0.50) | 24,436,369 | **0.8309** | **0.8937** | 7.18 |

---

## 모델 개선 및 정성 분석 메모
- **핵심 개선 포인트**: 픽셀 단위 이진 교차 엔트로피(`BCE`)와 이미지별 영역 겹침을 직접 최적화하는 `Per-Image Soft Dice Loss`를 결합하고, `CosineAnnealingLR` 및 `drop_last=True`를 적용해 학습 안정성과 경계 분할 정밀도를 확보함.
- **성공 및 취약 사례 분석**:
  - **고성능 사례**: 일반적 크기의 용종 및 주변 점막과 대비가 비교적 명확한 병변에서는 인코더-디코더의 Skip Connection을 통해 경계를 정밀하게 복원함(상위 샘플 Dice 0.96 이상).
  - **취약(실패) 사례**: 내시경 광원 반사가 강하게 겹친 영역이나 화면 대부분을 차지하는 초대형 병변, 주변 점막과 경계가 모호한 병변에서는 일부 영역이 누락되거나 경계가 뭉개지는 경향이 관찰됨.

---

## 산출물 및 체크포인트 경로
- **코드 경로 (`feature/unet`)**: `unet/config.json`, `unet/config.py`, `unet/train_unet.py`, `unet/evaluate.py`
- **결과 및 시각화 (`feature/unet`)**: `unet/outputs/unet_resnet34/`
  - `test_summary.json` (테스트 요약 지표)
  - `test_per_image_scores.csv` (테스트 150장 개별 IoU/Dice 점수)
  - `training_curves.png` (학습/검증 손실 및 지표 곡선)
  - `success_cases.png` (고성능 성공 사례 4종)
  - `failure_cases.png` (취약/실패 사례 4종)
- **학습 가중치 파일**: Google Drive `cv_project/checkpoints/unet_resnet34_bce_dice_best.pth`
