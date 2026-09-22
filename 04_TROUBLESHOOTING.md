# TROUBLESHOOTING — 안 될 때

`START_HERE.ipynb` 가 대부분을 잡아 주지만, 잡히지 않는 것과 해결 방법을 정리했습니다.

> ### 먼저 확인할 것
>
> ```bash
> pwd                                 # 패키지 루트인가?
> ls data/okm_augumented_2021.csv     # 이 파일이 보이는가?
> bash run.sh check                   # 환경 점검
> ```

---

## 1. 그림의 한글이 `□□□` 로 깨진다

**원인**: 한글 폰트가 없습니다. 코드의 폰트 폴백 체인은
`Malgun Gothic → NanumGothic → Noto Sans KR → AppleGothic` 인데,
`Malgun Gothic` 은 **Windows 전용**입니다. Linux pod 에는 보통 아무것도 없습니다.

```bash
sudo apt-get update && sudo apt-get install -y fonts-nanum
rm -rf ~/.cache/matplotlib        # ← 이 줄이 없으면 설치해도 계속 깨진다
```

그리고 **Jupyter 커널을 재시작**하세요. matplotlib 은 폰트 목록을 프로세스 시작 시
캐시하므로, 캐시를 지우고 커널을 다시 띄워야 반영됩니다.

**확인**: `START_HERE.ipynb` 3번 셀이 실제로 한글을 그려 glyph 누락 경고가 나는지
검사합니다. 설치 목록에 있어도 렌더가 안 되는 경우가 있어 렌더 테스트가 더 믿을 만합니다.

> root 권한이 없으면 `setup_pod.sh` 가 이 단계를 건너뛰고 경고만 냅니다.
> 폰트 없이도 파이프라인은 완주하며 **수치는 전부 정확합니다** — 그림의 한글만 깨집니다.

---

## 2. `pandas` 버전이 3.0.5 가 아니다

**원인**: pod 이미지에 pandas 2.x 가 선설치되어 있고, 다른 패키지가 그것을 붙들고 있습니다.

```bash
python -m pip install --no-cache-dir --force-reinstall "numpy==2.3.5" "pandas==3.0.5"
python -c "import pandas; print(pandas.__version__)"   # 3.0.5 여야 한다
```

`numpy` 를 함께 고정하는 것이 중요합니다. numpy ABI 가 어긋나면 pandas·scikit-learn·
lightgbm 이 import 단계에서 깨집니다.

**왜 3.0.5 를 고집하나**: 무음 실패 탐지가 pandas 3.x 의 Copy-on-Write 동작에 의존합니다.
가이드북 베이스라인의 연쇄 대입(`df['col'][mask] = 1`)을 `ChainedAssignmentError` 로
승격시켜 잡는데, 2.x 에서는 조용히 넘어갑니다. 또한 재현성 검증(722개 셀 일치)이
이 조합에서 이뤄졌습니다.

---

## 3. TensorFlow 경고가 쏟아진다 / CUDA 를 못 찾는다고 한다

**정상입니다.** 이 파이프라인은 **GPU 를 쓰지 않습니다.**

```
Could not find cuda drivers... GPU will not be used.
TF-TRT Warning: Could not find TensorRT
```

이런 메시지는 무해합니다. `s00_env.py` 가 `TF_CPP_MIN_LOG_LEVEL=3` 으로 억제하지만
일부는 import 시점에 이미 나옵니다. `run.sh finalize` 가 노트북에서 이 stderr 출력을 제거합니다.

**TensorFlow 설치 자체가 실패해도 괜찮습니다.** 코드에 미설치 방어가 들어 있어
SimpleRNN·DNN 2개 모델만 건너뛰고 나머지는 정상 실행됩니다.

```bash
python -c "import tensorflow" || echo "TF 없음 — 모델 2개만 건너뛰고 진행된다"
```

`tensorflow[and-cuda]` 는 설치하지 마세요. CUDA 라이브러리를 1GB 이상 끌어오는데
GPU 를 쓰지 않으므로 낭비입니다.

---

## 4. `FileNotFoundError: data/okm_augumented_2021.csv`

**원인**: 작업 디렉터리가 패키지 루트가 아닙니다. 노트북이 **상대경로**를 씁니다.

```bash
cd /workspace/KAMP_공모전      # 실제 경로로
ls data/okm_augumented_2021.csv
```

Jupyter 에서는 **이 폴더를 열고** 노트북을 실행하세요. 다른 폴더에서 파일만 열면
작업 디렉터리가 달라집니다.

노트북 안에서 확인:

```python
from pathlib import Path
print(Path.cwd(), Path("data/okm_augumented_2021.csv").exists())
```

---

## 5. 실행 중 세션이 끊겼다 / 브라우저를 닫았더니 멈췄다

**원인**: Jupyter UI 로 돌리면 커널이 프런트엔드 세션에 묶입니다.
FULL 실행은 CPU 에서 **30~60분**이라 이 사고가 가장 흔합니다.

**배치로 돌리세요**:

```bash
bash run.sh run
# 또는
nohup env PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8 \
  python -X utf8 -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=7200 자원최적화_제안모델.ipynb \
  > outputs/run.log 2>&1 &

tail -f outputs/run.log
```

진행 상황은 `outputs/tables/` 의 파일 타임스탬프로도 알 수 있습니다.

```bash
ls -lt outputs/tables/ | head        # 최근 갱신된 표가 현재 진행 중인 장
```

대략적인 순서: `ch1_*` → `ch2_feature_*` → `ch3_*` → `ch4_*`(4장 베이스라인) →
`hpo_trials`(5장, 가장 오래 걸림) → `ch2_regression`(6장) → `ch3_importance`(7장) →
`ch4_levers`(8장) → `ch6_*`(10장).

---

## 6. 너무 오래 걸린다 / 일단 구조만 확인하고 싶다

```bash
bash run.sh run-fast     # 3~5분
```

`KAMP_FAST=1` 로 Optuna 100회→6회, 에폭 500→5 로 축소합니다.

> ⚠️ **수치는 최종값이 아닙니다.** 보고서에 넣을 값은 반드시 `run.sh run`(FULL) 결과를 쓰세요.
> FAST 로 돌리면 `outputs/report_tbd_filled.md` 상단에 FAST 모드 경고가 찍힙니다.

FULL 과 FAST 의 차이:

| | FULL | FAST |
|---|---|---|
| Optuna 탐색 | 100회 | 6회 |
| LightGBM 트리 | 800 | 120 |
| RNN 에폭 | 500 | 5 |
| MLP 에폭 | 300 | 5 |
| 테스트 MAE | 5.220 | 5.353 |
| 소요 시간 | 30~60분 | 3~5분 |

---

## 7. 디스크가 부족하다

TensorFlow wheel 이 큽니다(약 600MB). `outputs/` 재생성은 약 10MB 입니다.

```bash
df -h .
python -m pip cache purge       # pip 캐시 정리
rm -rf outputs/figures outputs/tables   # 산출물만 삭제 (data/·src/ 는 보존)
```

---

## 8. 테스트가 실패한다

```bash
bash run.sh test                # 236개 통과가 정상
```

몇 가지 정상적인 경우:

- **1개 skipped** — 플레이스홀더 테스트입니다. 정상입니다.
- **1개 deselected** — `-m "not slow"` 로 제외한 Optuna 재탐색 테스트입니다.
  전부 돌리려면: `python -m pytest tests/ -q` (몇 분 더 걸림)

실제로 실패한다면:

```bash
# 버전 불일치가 가장 흔한 원인
python -m pytest tests/ -q -m "not slow" --tb=short 2>&1 | head -40
```

테스트는 실측 오라클을 검증합니다(예: `peak15` q95 = 187.0, 복제일 115일,
달력규칙 TP28/FN0/FP82/TN226). 패키지 버전이 다르면 이 값들이 미세하게 달라질 수 있습니다.

---

## 9. 노트북을 고쳤는데 다음 실행에서 사라졌다

**원인**: `src/sNN_*.py` 가 **단일 진실원**이고 노트북은 빌드 산출물입니다.
노트북을 다시 만들면(`run.sh build`) 덮어씁니다.

```bash
# 올바른 순서
vim src/s05_models.py        # 원본을 고친다
bash run.sh build            # 노트북 재조립 (159셀)
bash run.sh test             # 검증
bash run.sh run              # 실행 (finalize 자동)
```

---

## 10. 개인정보 스캔이 걸린다

```bash
bash run.sh finalize
```

이 단계가 **필수**입니다. 노트북 안(10.5절)의 스캔은 `nbconvert --inplace` 특성상
**자기 실행 출력을 볼 수 없습니다** — 모든 셀이 끝난 뒤에 파일이 쓰이기 때문입니다.
서드파티 경고가 stderr 로 내보내는 메시지에 설치 경로(사용자명 포함)가 박히는데,
블라인드 평가에서는 위반입니다.

`run.sh finalize` 가 stderr 출력을 제거하고 잔여 경로를 치환한 뒤 후스캔합니다.
0건이 아니면 어느 파일·셀인지 알려줍니다.

`_dev_docs/` 에는 개발 이력과 사용자명이 들어 있습니다. **제출 zip 에서 제외하세요.**

---

## 이 문서로 해결되지 않는 경우 — 알려진 한계

**Linux/pod 환경에서 실제 검증을 하지 못했습니다.** 이 패키지는 Windows 11 /
Python 3.11.9 에서 검증되었습니다. `setup_pod.sh` 는 문법 검사(`bash -n`)까지만
확인했고, 실제 pod 실행은 `START_HERE.ipynb` 의 점검 셀이 런타임에 잡도록 설계했습니다.

따라서 pod 에서 처음 돌릴 때는 **`bash run.sh check` 를 먼저 돌려** 점검 결과를 확인하세요.
점검이 통과하면 본실행도 통과할 가능성이 높습니다. 점검이 놓칠 수 있는 것:

- apt 저장소 접근이 막힌 폐쇄망 (폰트 설치 실패 → 한글만 깨짐, 수치는 정상)
- pip 저장소 접근 제한 (이 경우 오프라인 wheel 을 미리 준비해야 함)
- 매우 적은 vCPU (1~2코어)에서 LightGBM `num_threads=4` 가 오히려 느려질 수 있음
  — 재현성 때문에 고정한 값이라 바꾸면 수치가 달라집니다
