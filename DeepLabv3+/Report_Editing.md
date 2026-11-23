# 26-2 컴퓨터비전 미니 프로젝트 최종 보고서 (2팀)

> **팀원 안내 (보고서 미완성 초안)**
> - 현재 **DeepLabv3+ (강현구)** 파트의 실험 수치만 반영되어 있으며, 나머지 파트는 공란으로 비워 두었습니다.
> - **각자 맡은 파트의 실험 결과, 설정값, 시각화, 분석 내용을 아래 해당 공란에 직접 입력해 주세요.**
>   - **김장희(팀장)**: 최종 비교표 취합 및 검토
>   - **장정훈**: 고전 기법 튜닝 파라미터·테스트 결과, 실패 분석, CVC-ClinicDB 외부 평가 수치
>   - **강승태**: U-Net 학습 조건·파라미터 수·테스트 결과, 성공/실패 사례 합본 시각화

- **주제**: 대장 내시경 용종 분할 (고전 영상처리 vs U-Net vs DeepLabv3+)
- **팀원 및 역할**:
  - 김장희(팀장, 2021E7008): 데이터 분할, 공통 로더, 공통 평가 코드 및 최종 비교표
  - 장정훈(2021E8530): 고전 기법 구현·튜닝, 실패 분석 진행 및 CVC-ClinicDB 외부 평가
  - 강승태(2022E7001): U-Net 학습·추론, 성공·실패 사례 시각화 및 발표자료 취합
  - 강현구(2022E7002): DeepLabv3+ 학습·추론 및 최종 보고서 취합
- **GitHub 저장소**: https://github.com/OpenCV-Project/colonoscopy-polyp-segmentation

> 주의: 본 실험은 교육·연구 목적의 학습 실험이며, 임상적·진단적 성능을 주장하지 않습니다.

---

## 1. 문제 정의 및 데이터셋

### 1.1 목표
대장 내시경 영상에서 용종(Polyp) 영역을 픽셀 단위로 분할하고, 고전 영상처리 기법과 수업에서 학습한 딥러닝 모델(U-Net, DeepLabv3+)의 분할 성능(IoU, Dice)과 이미지당 처리 시간(ms)을 동일한 조건에서 비교한다.

### 1.2 데이터셋 및 고정 분할
- **데이터셋**: Kvasir-SEG (총 1,000장, RGB 내시경 이미지 + 이진 정답 마스크)
- **고정 분할 (`splits.csv`, seed=42)**:
  - 학습(train): 700장
  - 검증(val): 150장
  - 테스트(test): 150장
- **전처리 규격 (`dataset.py`)**:
  - 이미지: 256×256 리사이즈(`INTER_AREA`), RGB 변환
  - 마스크: 밝기 128 기준 이진화(0/1), 256×256 리사이즈(`INTER_NEAREST`)
  - 학습 증강: 확률 0.5 좌우 반전, ±10° 회전

---

## 2. 구현 방법 및 실험 조건

### 2.1 고전 영상처리 베이스라인 (담당: 장정훈)
- 방법 요약: 
- 검증 세트에서 확정한 파라미터 (군집 수, 면적 범위, 커널 크기 등): 

### 2.2 U-Net (담당: 강승태)
- 모델: `segmentation_models_pytorch.Unet` (`encoder_name="resnet34"`, `encoder_weights="imagenet"`)
- 학습 조건: 
- 체크포인트 선정 및 이진화 임계값: 
- 파라미터 수: 

### 2.3 DeepLabv3+ (담당: 강현구)
- 모델: `segmentation_models_pytorch.DeepLabV3Plus` (`encoder_name="resnet34"`, `encoder_weights="imagenet"`)
- 입력 정규화: ImageNet 평균 `(0.485, 0.456, 0.406)` / 표준편차 `(0.229, 0.224, 0.225)`
- 학습 조건: Batch size 16 (`drop_last=True`), 30 Epochs, AdamW (`lr=1e-4`, `weight_decay=1e-4`), CosineAnnealingLR, AMP(FP16)
- 체크포인트 선정: 검증 세트(`val`) 평균 Dice가 가장 높은 모델 저장 (Baseline BCE: Epoch 23 / val Dice 0.8844, 개선 BCE+Dice: Epoch 19 / val Dice 0.8872)
- 이진화 임계값: 검증 세트(`val`)에서만 최적 임계값 선택 (Baseline BCE: `0.45`, 개선 BCE+Dice: `0.40`)
- 파라미터 수: `22,437,457` (전체 학습 가능, 약 22.44M)

---

## 3. 정량 평가 결과 (Kvasir-SEG 테스트 150장)

> `metrics.py` 및 `benchmark.py` 기준, 동일 장비에서 측정한 결과를 각자 기입해 주세요.

| 방법 | 실행 장치 | 전·후처리 범위 | 파라미터 수 | Mean IoU | Mean Dice | 처리 시간 (ms/img) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| 고전 기법 (Lab/HSV + K-means) | | | - | | | |
| U-Net (ResNet-34) | | | | | | |
| DeepLabv3+ (ResNet-34, Baseline BCE) | Colab CUDA GPU | 리사이즈, ImageNet 정규화, 이진화(0.45) | 22,437,457 | 0.8041 | 0.8758 | 13.18 |
| DeepLabv3+ (ResNet-34, BCE+Dice) | Colab CUDA GPU | 리사이즈, ImageNet 정규화, 이진화(0.40) | 22,437,457 | **0.8262** | **0.8937** | **11.52** |

---

## 4. 정성 평가 (성공 사례 및 실패 사례)

> 원본 / 정답 / 고전 기법 / U-Net / DeepLabv3+ 예측 결과를 나란히 배치해 주세요.

### 4.1 성공 사례 (3장 이상)
- DeepLabv3+ (BCE+Dice) 기준 상위 3장 시각화: `DeepLabv3+/outputs/improved_bce_dice/success_cases.png` (Baseline: `DeepLabv3+/outputs/baseline_bce/success_cases.png`)
- 합본 그림 및 분석 (담당자 기입): 

### 4.2 실패 사례 (3장 이상)
- DeepLabv3+ (BCE+Dice) 기준 하위 3장 시각화: `DeepLabv3+/outputs/improved_bce_dice/failure_cases.png` (Baseline: `DeepLabv3+/outputs/baseline_bce/failure_cases.png`)
- 합본 그림 및 분석 (담당자 기입): 

---

## 5. 실패 분석 및 개선 실험

### 5.1 오분할 원인 분석
- **고전 기법 (장정훈 기입)**: 
- **U-Net (강승태 기입)**: 
- **DeepLabv3+ (강현구)**: `BCEWithLogitsLoss` 단독 학습 시 배경 픽셀 비중이 큰 소형 용종에서 경계가 뭉개지거나 과소 분할되는 경향 확인.

### 5.2 개선 가설 및 실험 결과
- **DeepLabv3+ 개선 가설**: 픽셀 단위 손실(`BCE`)에 영역 겹침을 직접 최적화하는 `Soft Dice Loss`를 결합(`BCEDiceLoss`)하고, 검증 세트에서 최적 임계값을 조정하면 소형 용종 및 경계 분할 성능이 향상될 것이다.
- **기타 모델 개선 내용 (담당자 기입)**: 

| 모델 설정 | 검증 최적 임계값 | Test IoU | Test Dice | ms/img | 비고 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 고전 기법 (개선 전 / 후) | | | | | |
| U-Net (개선 전 / 후) | | | | | |
| DeepLabv3+ (개선 전: BCE) | 0.45 | 0.8041 | 0.8758 | 13.18 | 베이스라인 (Epoch 23) |
| DeepLabv3+ (개선 후: BCE+Dice) | 0.40 | **0.8262** (+0.0221) | **0.8937** (+0.0179) | 11.52 | 개선 적용 (Epoch 19) |

---

## 6. 확장 실험: CVC-ClinicDB 외부 일반화 평가

Kvasir-SEG에서 학습·확정한 세 방법을 외부 공개 데이터셋인 CVC-ClinicDB에 추가 학습 없이 그대로 적용해 일반화 성능을 비교한다.

| 방법 | Kvasir-SEG Test IoU | Kvasir-SEG Test Dice | CVC-ClinicDB IoU | CVC-ClinicDB Dice |
| :--- | :---: | :---: | :---: | :---: |
| 고전 기법 | | | | |
| U-Net (ResNet-34) | | | | |
| DeepLabv3+ (ResNet-34, BCE+Dice) | 0.8262 | 0.8937 | | |

---

## 7. 결론 및 한계점
- (전체 팀원 결과 취합 후 작성 예정)

---

## 8. 참고문헌 및 도구 사용 내역

### 8.1 데이터셋 및 라이브러리 인용
- **Kvasir-SEG**: Jha, D., et al. "Kvasir-SEG: A Segmented Polyp Dataset." *MultiMedia Modeling (MMM)*, 2020.
- **CVC-ClinicDB**: Bernal, J., et al. "WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians." *Computerized Medical Imaging and Graphics*, 2015.
- **DeepLabv3+**: Chen, L.-C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." *ECCV*, 2018.
- **segmentation_models_pytorch**: Iakubovskii, P. (2019), GitHub repository.

### 8.2 생성형 AI 및 외부 코드 사용 내역
- **사용 도구**: DeepLabv3+ 학습·평가 파이프라인 및 보고서 초안 작성 보조
- **외부 코드 출처**: `segmentation_models_pytorch` (`DeepLabV3Plus`, `Unet`) 공식 API 활용
