# colonoscopy-polyp-segmentation

Kvasir-SEG 대장 내시경 용종 분할을 대상으로 **고전 영상처리(Lab·HSV + K-means + 모폴로지)**, **U-Net**, **DeepLabv3+**의 성능을 동일한 테스트 세트에서 공정하게 비교하는 26-2 컴퓨터비전 미니 프로젝트(2팀) 저장소입니다.

> 본 프로젝트는 교육·연구 목적의 학습 실험이며, 임상적·진단적 성능을 주장하지 않습니다.

---

## 1. 팀원 및 역할 분담

| 담당자 | 역할 |
| :--- | :--- |
| **김장희 (팀장)** | 데이터 분할, 공통 로더, 공통 평가 코드 및 최종 비교표 |
| **장정훈** | 고전 기법 구현·튜닝, 실패 분석 진행 및 CVC-ClinicDB 외부 평가 |
| **강승태** | U-Net 학습·추론, 성공·실패 사례 시각화 및 발표자료 취합 |
| **강현구** | DeepLabv3+ 학습·추론 및 최종 보고서 취합 |

---

## 2. 저장소 구조

```text
colonoscopy-polyp-segmentation/
├── README.md                  # 전체 프로젝트 안내 및 실행 가이드
├── AGENTS.md                  # 팀 공통 실험·코드 규칙
├── .gitignore                 # 데이터셋·모델 가중치(*.pth)·임시 파일 제외
├── split_dataset.py           # Kvasir-SEG 고정 분할(seed=42, 700/150/150) 생성
├── splits.csv                 # 고정된 학습/검증/테스트 파일 목록
├── dataset.py                 # 256×256 리사이즈 및 RGB/이진 마스크 로더
├── metrics.py                 # 이미지별 IoU·Dice 및 산술 평균 계산
├── benchmark.py               # 예열 후 이미지당 평균 처리 시간(ms) 측정
├── train_unet.py              # U-Net (ResNet-34) 학습 및 평가 스크립트
└── DeepLabv3+/
    ├── config.py              # DeepLabv3+ 학습·평가 하이퍼파라미터 설정
    ├── model.py               # DeepLabv3+ 모델 생성, ImageNet 정규화, BCEDiceLoss
    ├── train.py               # 학습 루프, 체크포인트 저장, 검증 세트 임계값 튜닝
    ├── evaluate.py            # 테스트 세트 평가, 처리 시간 측정, 시각화, CVC-ClinicDB 평가
    ├── run_colab.py           # Colab/CLI 원클릭 학습·개선실험·평가 실행 스크립트
    ├── Report_Editing.md      # 2팀 최종 보고서 취합 초안 (팀원별 수치 기입용)
    └── outputs/               # DeepLabv3+ 실험 결과(CSV, JSON, 성공/실패 PNG)
```

---

## 3. 데이터셋 준비

> 원본 데이터셋과 모델 가중치(`*.pth`)는 GitHub에 업로드하지 않습니다.

1. **Kvasir-SEG**: [공식 배포 페이지](https://datasets.simula.no/kvasir-seg/)에서 `kvasir-seg.zip`을 내려받아 저장소 외부(예: Google Drive `MyDrive/Kvasir-SEG`)에 압축을 풉니다.
   - `images/`: 내시경 원본 이미지 (`.jpg`)
   - `masks/`: 동일 파일명의 정답 마스크 (`.jpg`)
2. **고정 분할 (`splits.csv`)**: 시드 `42`로 학습 700장 / 검증 150장 / 테스트 150장으로 분할되어 있으며, 모든 모델은 이 분할을 동일하게 사용합니다.
   - 이미지: `256×256` 리사이즈 (`cv2.INTER_AREA`), RGB 변환
   - 마스크: 밝기 `128` 이상 이진화(`0`/`1`), `256×256` 리사이즈 (`cv2.INTER_NEAREST`)
3. **CVC-ClinicDB (확장 평가용)**: 외부 일반화 성능 비교 시 추가 학습·재튜닝 없이 그대로 적용합니다.

---

## 4. 실행 방법

### 4.1 필수 패키지 설치
```bash
pip install torch torchvision opencv-python numpy pandas matplotlib segmentation-models-pytorch
```

### 4.2 DeepLabv3+ 학습 및 평가 (Google Colab / CLI)
`run_colab.py`는 베이스라인(`BCEWithLogitsLoss`)과 개선 실험(`BCEDiceLoss`)을 순차 실행하고, 검증 세트(`val`)에서 최적 임계값을 찾은 뒤 테스트 세트(150장) 평가·벤치마크·시각화를 자동 저장합니다.

```bash
python DeepLabv3+/run_colab.py \
  --project-root "/경로/프로젝트_루트" \
  --epochs 30 \
  --batch-size 16
```

- 빠른 동작 확인: `--epochs 2 --skip-improved` 옵션 사용
- 외부 데이터셋 평가 포함: `--cvc-dir "/경로/CVC-ClinicDB"` 옵션 추가
- 결과 저장 위치: `DeepLabv3+/outputs/` (`deeplabv3plus_comparison.csv`, `test_summary.json`, `success_cases.png`, `failure_cases.png`)

### 4.3 U-Net 학습 및 평가
```bash
python train_unet.py
```

---

## 5. 공정 비교 및 작업 규칙 요약

- **테스트 세트 격리**: 하이퍼파라미터, 고전 기법 파라미터, 이진화 임계값, 체크포인트 선택은 모두 **검증 세트(`val`)**에서만 수행합니다.
- **공통 모듈 사용**: 데이터 로딩(`dataset.py`), 지표 계산(`metrics.py`), 처리 시간 측정(`benchmark.py`)은 팀 공통 코드를 사용합니다.
- **최종 보고서 취합**: 각 팀원은 본인 파트의 실험 결과를 `DeepLabv3+/Report_Editing.md`의 해당 공란에 기입한 뒤 PR로 공유합니다.

---

## 6. 참고문헌

- **Kvasir-SEG**: Jha, D., et al. "Kvasir-SEG: A Segmented Polyp Dataset." *MMM*, 2020.
- **CVC-ClinicDB**: Bernal, J., et al. "WM-DOVA maps for accurate polyp highlighting in colonoscopy." *Computerized Medical Imaging and Graphics*, 2015.
- **DeepLabv3+**: Chen, L.-C., et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation." *ECCV*, 2018.
- **segmentation_models_pytorch**: Iakubovskii, P. (2019), GitHub repository.
