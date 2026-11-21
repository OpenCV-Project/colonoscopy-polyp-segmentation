---
name: code-fundamentals
description: 가독성, 예측 가능성, 응집도, 결합도 4가지 기준으로 Python 코드를 작성하고 리뷰하는 코드 품질 가이드. 이 레포에서 새 코드 작성, 리팩토링, 코드 리뷰를 할 때, 또는 "코드 품질", "클린 코드", "코드 개선" 요청 시 사용한다.
---

# code-fundamentals

> 좋은 코드는 **변경하기 쉬운** 코드다.

출처: [Toss Frontend Fundamentals](https://frontend-fundamentals.com) ([GitHub](https://github.com/toss/frontend-fundamentals), MIT License). 원본에서 언어와 상관없이 통하는 규칙만 골라 Python 예시로 옮겼다. React·Hook·폼·Props Drilling처럼 프론트엔드에만 해당하는 규칙은 뺐다.

## 사용 방법

- **코드 작성**: 아래 기준을 순서대로 지키며 쓴다.
- **코드 리뷰**: 기준 순서대로 검토하고, 위반한 곳마다 `파일:줄`, 규칙 이름, 고친 코드 예시를 함께 제시한다.
- 이 레포의 `AGENTS.md` 규칙(공통 코드 사용, 경로 하드코딩 금지, 읽기 쉬운 코드 우선)이 이 가이드보다 우선한다.

## 우선순위와 트레이드오프

**가독성 > 예측 가능성 > 응집도 > 결합도**

- 코드는 쓰는 시간보다 읽는 시간이 훨씬 길다. 기준이 충돌하면 앞의 것을 우선한다.
- 정답이 분명하지 않으면 변경 빈도, 이해 비용, 버그 위험을 근거로 판단하고 그 근거를 적는다.
- 항상 함께 바뀌는 코드는 응집도를, 따로 바뀌는 코드는 결합도를 우선한다.
- 읽기 어려워지는 추상화보다 다소 중복이 있어도 읽기 쉬운 코드를 고른다.

## 1. 가독성

### 같이 실행되지 않는 코드 분리하기

동시에 실행되지 않는 분기를 한 함수에 섞지 말고 경우별 함수로 나눈다.

```python
# Bad: 방법마다 다른 흐름이 한 함수에 섞여 있다
def predict(image, method):
    if method == "classic":
        ...  # 색공간 변환, K-means, 모폴로지
    else:
        ...  # 텐서 변환, 모델 추론, 이진화

# Good
def predict_classic(image): ...
def predict_deep(image, model): ...
```

### 구현 상세 추상화하기

저수준 처리는 의미 있는 함수 이름 뒤에 두어 "무엇을 하는지"가 먼저 읽히게 한다.

```python
# Bad
mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)

# Good
mask = refine_mask(mask, kernel)
```

### 로직 종류에 따라 함수 쪼개기

데이터 가공, 파일 저장·출력 같은 부수 효과, 계산을 한 함수에 섞지 않는다.

```python
# Bad: 지표 계산 함수가 결과 저장까지 한다
def evaluate(preds, masks, out_path):
    scores = [dice(p, m) for p, m in zip(preds, masks)]
    pd.DataFrame(scores).to_csv(out_path)
    return np.mean(scores)

# Good
scores = compute_dice_scores(preds, masks)
save_scores(scores, out_path)
```

### 복잡한 조건에 이름 붙이기

여러 조건이 얽히면 `is_*`, `has_*`, `should_*` 이름을 붙인다.

```python
# Bad
if 0.01 < area / total < 0.6 and mean_a == max_a:

# Good
is_valid_area = MIN_AREA_RATIO < area / total < MAX_AREA_RATIO
is_reddest_cluster = mean_a == max_a
if is_valid_area and is_reddest_cluster:
```

### 매직 넘버에 이름 붙이기

의미 있는 숫자는 이름을 가진 상수로 바꾼다.

```python
# Bad
binary = prob > 0.5

# Good
POLYP_THRESHOLD = 0.5
binary = prob > POLYP_THRESHOLD
```

### 시점 이동 줄이기

읽는 사람이 위아래로 오가지 않도록 변수는 쓰는 곳 가까이에 선언하고, 짧은 로직은 쓰는 자리에서 바로 보이게 한다.

```python
# Bad: 앞에서 만든 값을 한참 뒤에서 쓴다
start = time.perf_counter()
image = load_image(path)
...  # 30줄
elapsed_ms = (time.perf_counter() - start) * 1000

# Good: 측정 범위를 한눈에 보이게 묶는다
start = time.perf_counter()
mask = predict(image)
elapsed_ms = (time.perf_counter() - start) * 1000
```

### 조건 표현식 단순하게 하기

중첩된 조건 표현식(`a if x else b if y else c`)은 `if` 문으로 푼다.

```python
# Bad
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Good
def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

### 왼쪽에서 오른쪽으로 읽히게 하기

비교는 변수를 왼쪽에 두고, 범위는 수학식처럼 쓴다. 깊은 중첩은 조기 반환으로 편다.

```python
# Bad
if 0.5 <= prob: ...
if x >= low and x <= high: ...

# Good
if prob >= 0.5: ...
if low <= x <= high: ...
```

## 2. 예측 가능성

### 이름 겹치지 않게 관리하기

다른 역할에는 다른 이름을 쓰고, 같은 역할에는 레포 전체에서 같은 이름을 쓴다.

```python
# Bad: 둘 다 load_data인데 하나는 경로 목록, 하나는 배열을 돌려준다
from data.split import load_data
from data.dataset import load_data

# Good
from data.split import load_split_paths
from data.dataset import load_images
```

### 같은 종류의 함수는 반환 타입 통일하기

같은 역할의 함수는 같은 형태를 반환하고, 실패도 같은 방식으로 처리한다.

```python
# Bad: 고전 기법은 0/1 uint8, 딥러닝은 0~1 float 확률을 반환한다
def predict_classic(image) -> np.ndarray: ...  # uint8 {0, 1}
def predict_deep(image, model) -> np.ndarray: ...  # float [0, 1]

# Good: 둘 다 같은 형태의 이진 마스크를 반환한다
def predict_classic(image) -> np.ndarray: ...  # uint8 {0, 1}, (H, W)
def predict_deep(image, model) -> np.ndarray: ...  # uint8 {0, 1}, (H, W)
```

### 숨은 로직 드러내기

함수는 이름이 말하는 일만 한다. 숨은 부수 효과가 필요하면 호출하는 쪽에서 드러내거나 이름에 반영한다.

```python
# Bad: 평가 함수가 몰래 체크포인트를 저장한다
def evaluate(model, loader):
    score = ...
    torch.save(model.state_dict(), "best.pth")
    return score

# Good
score = evaluate(model, loader)
if score > best_score:
    save_checkpoint(model, ckpt_dir)
```

## 3. 응집도

### 함께 수정되는 파일을 같은 디렉터리에 두기

종류(모든 모델, 모든 유틸)가 아니라 기능(고전 기법, 딥러닝, 평가) 기준으로 묶는다. 함께 바뀌는 코드가 같은 곳에 있으면 고칠 때 빠뜨리지 않는다.

```text
# Bad
utils/  models/  constants/

# Good
classic/   # 고전 기법 코드와 그 파라미터
deep/      # 모델 생성, 학습
evaluation/  # 지표, 처리 시간 측정
```

### 상수는 쓰는 곳 가까이 두기

한 파일에서만 쓰면 그 파일 위쪽에, 한 기능 안에서 공통이면 그 기능 폴더에 둔다. 모든 상수를 전역 파일 하나에 모으지 않는다. 실험 조건(에폭, 학습률, 임계값 등)은 상수가 아니라 공통 설정 파일에 둔다.

## 4. 결합도

### 책임을 하나씩 관리하기

함수와 모듈은 한 가지 일만 하고, 필요한 것만 import한다. 이름에 `and`가 필요하면 나눌 때가 됐다는 신호다.

```python
# Bad
def load_and_train_and_evaluate(config): ...

# Good
loaders = build_loaders(config)
model = train(config, loaders)
scores = evaluate(model, loaders.test)
```

### 중복 코드 허용하기

서로 다른 맥락에서 우연히 비슷해 보이는 코드는 성급하게 공통화하지 않는다. 세 번 이상 반복되고(Rule of Three) 같은 규칙이라 한 번에 같이 바뀌어야 할 때만 합친다. 합치려다 옵션과 분기가 늘어나면 중복을 그대로 둔다.

```python
# Bad: 두 모델의 차이를 옵션으로 흡수하려다 분기가 늘어난다
def build_model(name, use_aspp=False, decoder_channels=None, ...): ...

# Good: segmentation_models_pytorch 호출을 그대로 둔다
unet = smp.Unet("resnet34", encoder_weights="imagenet", classes=1)
deeplab = smp.DeepLabV3Plus("resnet34", encoder_weights="imagenet", classes=1)
```

단, `AGENTS.md`에서 공통 코드로 정한 데이터 로딩·전처리·지표 계산은 우연한 중복이 아니라 같은 규칙이므로 반드시 공통 코드를 쓴다.
