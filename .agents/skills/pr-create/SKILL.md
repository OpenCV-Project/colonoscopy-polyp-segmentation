---
name: pr-create
description: gh CLI로 현재 브랜치의 커밋과 diff를 분석해, 레포의 PR 템플릿(.github/pull_request_template.md)에 맞춘 한국어 PR 본문을 쓰고 브랜치를 푸시해 PR을 생성하거나 업데이트한다. PR 설명만 요청하면 본문만 보여 주고 멈춘다. "PR 만들어줘", "PR 설명 작성", "PR description 써줘", "PR 올려야 하는데", "pr-create" 등의 요청에 사용한다.
---

# pr-create

커밋과 diff를 분석해 `.github/pull_request_template.md` 구조 그대로 한국어 PR 본문을 만들고, `gh`로 PR을 생성하거나 업데이트한다.

## 중요 규칙

1. 커밋되지 않은 변경 사항은 커밋하거나 스테이징하지 않는다.
2. PR은 머지하지 않는다.
3. PR 제목과 본문은 한국어로 쓴다.
4. 본문 끝에 도구 홍보 문구(예: "made by claude", "Generated with ...")를 붙이지 않는다.
5. `AGENTS.md`의 표현 규칙을 따른다. 진단 성능을 주장하는 표현을 쓰지 않는다.
6. 사용자가 PR 설명(본문)만 요청했으면 Step 6까지만 하고, 제목과 본문을 보여 준 뒤 멈춘다. 푸시하거나 PR을 만들지 않는다.
7. 셸 변수는 명령 사이에 유지되지 않는다. `CURRENT_BRANCH`, `PR_TITLE`, `<번호>`는 앞 단계에서 얻은 실제 값으로 바꿔 명령에 직접 넣는다.
8. 아래 단계에서 "PR을 올리지 않는다"고 한 경우에는 푸시와 PR 생성·수정을 하지 않고, 사용자에게 다음 형식으로 알린다. 본문을 이미 썼다면 함께 보여 준다.

   > {문제} 때문에 PR은 아직 올리지 않았어요. {해결 방법}을 한 뒤 다시 요청해 주세요.

## Step 1: 전제 조건 확인

```bash
gh --version
gh auth status
git rev-parse --git-dir
git status --porcelain --untracked-files=all
```

- `gh`가 없으면 중단하고 설치를 안내한다 (Mac: `brew install gh`, Windows: `winget install GitHub.cli`, 설치 후 `gh auth login`).
- 인증이 안 되어 있으면 중단하고 `gh auth login`을 안내한다.
- `git status` 결과가 비어 있지 않으면 계속 진행하되, 커밋되지 않은 파일 목록과 "이 변경은 PR에 포함되지 않는다"는 점을 사용자에게 알린다.

## Step 2: 브랜치 정보

```bash
git rev-parse --abbrev-ref HEAD
```

- 결과를 `CURRENT_BRANCH`로 둔다. `main`이면 중단한다 (`main`에 직접 올리지 않는다).
- 베이스 브랜치는 `main`이다.
- 브랜치 이름은 `{type}/{설명}` 형식이다 (예: `init/harness-setting/#2`, `feat/data-loader`). 첫 `/` 앞을 `PR_TYPE`으로 쓴다. `/`가 없으면 Step 5의 추론 규칙을 따른다.

## Step 3: 원격 상태 확인

```bash
git fetch origin
git rev-parse --verify --quiet "refs/remotes/origin/${CURRENT_BRANCH}"
git rev-list --left-right --count "origin/${CURRENT_BRANCH}...HEAD"
```

- `git fetch`가 실패하면 마지막으로 받아 둔 `origin/main` 기준으로 분석은 계속하되, **PR을 올리지 않는다** (문제: 원격 정보를 받아오지 못함 / 해결: 네트워크나 인증을 확인).
- 원격 브랜치가 없으면(`rev-parse` 결과 없음) Step 7에서 새로 푸시한다.
- 원격 브랜치가 있으면 `rev-list` 결과 `{원격에만 있는 커밋 수} {로컬에만 있는 커밋 수}`를 본다.
  - 원격에만 있는 커밋이 있으면(첫 숫자 > 0) **PR을 올리지 않는다** (문제: 원격 브랜치에 로컬에 없는 커밋이 있음 / 해결: `git pull`로 원격 커밋을 받아 정리).

## Step 4: 변경 사항 분석

```bash
git log origin/main..HEAD --oneline
git diff origin/main...HEAD --name-status
git diff origin/main...HEAD
```

- 변경 파일이 30개 이상이면 전체 diff 대신 `--stat`, 커밋 목록, 핵심 경로의 부분 diff로 분석하고, 본문에 요약이 파일·커밋 단위라는 점을 짧게 밝힌다.

## Step 5: PR 제목

형식: `{PR_TYPE}: {한 줄 요약}` (예: `init: AI 작업 규칙 문서 추가`)

브랜치에서 타입을 못 얻으면 커밋 첫 줄의 접두사(`feat`, `fix`, `refactor`, `init`, `chore`, `docs`) 중 등장한 것을 이 순서로 우선해 고른다. 하나도 없으면 `chore`.

## Step 6: PR 본문

`.github/pull_request_template.md`를 읽고 섹션 순서와 제목을 그대로 쓴다. 템플릿의 `<!-- -->` 안내 주석은 지운다.

### 담당 파트

- 템플릿의 체크리스트를 그대로 두고, 변경 내용에 해당하는 항목만 `[x]`로 바꾼다. 항목을 추가하거나 지우지 않는다.

### 작업 내용

- 파일 단위가 아니라 기능·의도 단위로 적는다.
- 실험 관련 변경이면 `AGENTS.md`의 규칙과 맞닿는 부분(데이터 분할, 검증/테스트 사용, 공통 설정)을 짧게 적는다.
- **결정 사항 (있을 때만)**: 이 대화에서 사용자와 함께 고른 것과 고려했다가 배제한 것을 이유와 함께 적는다. 아래처럼 하위 목록으로 둔다.

  ```markdown
  - 결정 사항
    - 선택: 공통 설정 파일 하나 + CLI 인자로 경로 덮어쓰기 (실행 환경마다 경로가 달라서)
    - 배제: 환경 변수로 경로 전달 (어떤 값이 쓰였는지 결과에 남기기 어려워서)
  ```

  - 대화에 실제로 있었던 결정만 적는다. diff만 보고 추측해서 만들지 않는다.
  - 대화 맥락이 없거나 의미 있는 결정이 없으면 이 항목을 통째로 뺀다.

### AI 활용

- 이 세션에서 AI가 한 작업을 적는다 (예: 코드 초안 작성, 문서 정리, 리뷰).
- 외부 코드를 참고했다면 출처 URL도 적는다.
- 사람이 직접 한 부분과 구분이 어려우면 `[작성 필요]`로 남긴다.

본문은 레포 루트의 `.pr-body.md`에 저장한다 (`.gitignore`에 등록되어 있어 커밋되지 않는다).

설명만 요청받았거나 Step 3에서 PR을 올리지 않기로 했다면 여기서 멈추고 제목과 본문을 보여 준다.

## Step 7: 푸시

- 원격 브랜치가 없으면 `git push -u origin "${CURRENT_BRANCH}"`
- 로컬에만 있는 커밋이 있으면(Step 3 `rev-list`의 두 번째 숫자 > 0) `git push origin "${CURRENT_BRANCH}"`
- 푸시가 실패하면 **PR을 올리지 않는다** (문제: 푸시 실패와 에러 요약 / 해결: 에러에 맞는 조치).

## Step 8: 생성 또는 업데이트

```bash
gh pr list --head "${CURRENT_BRANCH}" --base main --json number,url,state
```

PR이 없으면:

```bash
gh pr create --base main --head "${CURRENT_BRANCH}" --title "${PR_TITLE}" --body-file .pr-body.md
```

PR이 있으면:

- `gh pr edit --body-file`은 기존 본문을 전부 바꾼다. `gh pr view <번호> --json body`로 확인해 본문이 비어 있지 않으면 교체할지 사용자에게 먼저 묻는다.
- 교체하기로 하면 `gh pr edit <번호> --title "${PR_TITLE}" --body-file .pr-body.md`
- 교체하지 않기로 하면 `gh pr edit`을 호출하지 않는다.

끝나면 `.pr-body.md`를 지우고 PR URL을 보여 준다. PR 생성·수정이 실패하면 **PR을 올리지 않았다**고 알리고(문제: 에러 요약 / 해결: 에러에 맞는 조치), 본문을 보여 줘 수동으로 만들 수 있게 한다.
