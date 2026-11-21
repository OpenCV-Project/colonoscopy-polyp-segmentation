# colonoscopy-polyp-segmentation
Kvasir-SEG 대장 내시경 용종 분할: 고전 영상처리(K-means, 모폴로지)와 U-Net, DeepLabv3+ 비교

##데이터 준비
1. [Kvasir-SEG 공식 배포 페이지](https://datasets.simula.no/kvasir-seg/)에서 `kvasir-seg.zip`을 내려받아 압축을 풉니다.
2. 압축을 푼 `Kvasir-SEG` 폴더에서 `images/`는 내시경 이미지, `masks/`는 같은 파일명의 정답 마스크입니다.
3. 팀 공통 `splits.csv`를 사용합니다. 학습 700쌍, 검증 150쌍, 테스트 150쌍으로 구성되며, 테스트 세트는 최종 평가에 사용합니다.

## 공통 코드 설명
- `split_dataset.py` — 시드 42로 데이터 분할 목록 생성
- `dataset.py` — 이미지와 정답 마스크를 256×256으로 불러오기
- `metrics.py` — 이미지별 IoU·Dice와 전체 평균 계산
- `benchmark.py` — 예열 후 이미지당 평균 처리 시간 측정
