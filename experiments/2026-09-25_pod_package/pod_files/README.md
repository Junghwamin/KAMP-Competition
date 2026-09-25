# 자원최적화 제안모델 — 실험 전용 단일 노트북 (포드 업로드용)

노트북 1개와 데이터 1개로 처음부터 끝까지 돌아간다.
**이전 버전(325셀)은 KAMP 포드(JupyterHub)에서 실제로 완주 확인했다** — 아래 10절 참고.
이번 버전(361셀)은 모델 저장(10.5절)과 지연변수 제거 효과 분해(6.6절)가 추가됐다. 개발 PC 에서 FULL 완주를 확인했고,
**포드에서는 다시 실행해야 한다**(명령은 아래 그대로).

## 빠른 시작 (포드에서 실제로 통한 순서)

```bash
cd ~/pod_실험전용        # 노트북과 data/ 가 있는 폴더

# 1) 진단 — 읽기만 한다
bash check_env.sh

# 2) 설치 — 없는 것만, 기존 버전은 건드리지 않고
bash setup_pod.sh

# 3) 빠른 확인 (10분 내외) → 통과하면 본실행
KAMP_FAST=1 bash run.sh
nohup bash run.sh > full_run.out 2>&1 &   # FULL 45~75분, 세션 끊겨도 안전
tail -f full_run.out                       # Ctrl+C 로 보기만 중단
```

> ⚠️ **`setup_pod.sh` 를 그냥 돌렸다가 `Permission denied` 로 막힌 적이 있다.**
> 원인과 해결은 **9절**에 적어 뒀다. 지금 스크립트는 그 문제를 고친 버전이지만,
> 포드에서 검증된 것은 9절의 **수동 3줄**이다. 스크립트가 막히면 9절대로 하면 된다.

주피터 UI 로 돌릴 거면 2)까지 하고 `자원최적화_실험전용.ipynb` 를 열어
**위에서 아래로 전부 실행**한다. 단, 6절(재현성)을 먼저 읽을 것.

---

## 1. 폴더 구성

```
.
├── 자원최적화_실험전용.ipynb   ← 이것 하나가 전부다 (출력 없는 깨끗한 상태, 361셀)
├── data/
│   └── okm_augumented_2021.csv   ← 경로·파일명 그대로 둘 것
├── check_env.sh                ← 환경 진단 (읽기 전용, 아무것도 안 고침)
├── setup_pod.sh                ← 없는 패키지만 설치 + 한글폰트
├── run.sh                      ← 배치 실행 + 완주 검사
├── requirements.txt            ← 참고용. 공유 환경에서 그대로 쓰면 안 된다(9절)
├── serving/                    ← (선택) 저장된 모델로 익일 예측 — 12절. 노트북 실행에는 필요 없다
└── README.md
```

실행하면 같은 폴더 밑에 `outputs/` 가 생긴다. 미리 만들어 둘 필요 없다.

**경로 주의** — CSV 는 노트북 옆이 아니라 `data/` 밑에 있어야 한다.
코드가 `Path("data") / "okm_augumented_2021.csv"` 를 상대경로로 읽고,
`data` 폴더는 자동 생성되지 않는다. 작업 디렉터리(CWD)도 이 폴더여야 한다.

## 2. 노트북 구조

| | 내용 |
|---|---|
| **1부 · 환경과 정의** | 임포트 · 경로 · 시드 · 한글폰트 · 저장 유틸 + **함수 정의 전부**. 여기서는 실험을 하지 않는다. |
| **2부 · 실험** | 데이터 적재 → 진단 → 파생변수 → 분할 → 베이스라인 → 제안모델 → 성능평가 → 영향요인·오류분석 → 피크저감 시뮬레이션 → 테스트 예측결과 파일 |

`src/*.py` 에서 자동 조립된 빌드 산출물이다(생성기 `tools/build_experiment_notebook.py`).
**보고서·제출물 생성 8개 절은 빠져 있다** — 보고서 서술 교정(6.7), 보고서 4·5장 초안(8.5·9장), 채움표,
requirements/README 생성, 발표자료 pptx·개인정보 스캔(10.6), 제출 준비 게이트.
실험 결과(모델·지표·그림·표·예측파일)에는 영향이 없다. **모델 저장(10.5)과 6.6절은 포함된다.**

## 3. 읽고 쓰는 파일

- **읽는 파일: `data/okm_augumented_2021.csv` 단 하나.** 그 외에는 아무것도 읽지 않는다(10.5절이 방금 저장한 모델 번들을 다시 읽어 검증하는 것 제외).
- 쓰는 파일: `outputs/` 밑의 **그림 39장**(F01~F39) · **표 106개** · 그림 색인 1개
  (`outputs/figures/figure_index.csv`) · 예측결과 1개(`outputs/predictions_test_336h.csv`).
  **모델 번들**: FULL 이면 `outputs/models/full/{eval,deploy}/`, FAST 면 `outputs/models/fast/...` (각 약 25MB).
  eval = 제출 예측을 만든 모델(9/1 이전 학습), deploy = 같은 방식으로 9/14 까지 재학습한 배포용.
  노트북이 저장 직후 다시 읽어 예측이 비트 단위로 같은지, 재학습 해시가 같은지 확인하고 어긋나면 멈춘다.
  `outputs/baseline_repro` 는 만들어지지만 비어 있다.

## 4. 필요한 패키지

노트북이 실제로 임포트하는 것은 10개다:

```
numpy pandas scipy scikit-learn matplotlib lightgbm optuna statsmodels tensorflow shap
```

**KAMP 포드에는 이 중 8개가 이미 있었고 `optuna` 와 `shap` 두 개만 없었다.**
`requirements.txt` 는 개발 환경의 버전을 적어둔 **참고용**이다.
**공유 환경에서 그대로 설치하면 안 된다** — 9절을 볼 것.

### 실제 최소 버전 (코드 근거 + 실측)

여러 버전을 실제로 깔아 돌려 확인한 하한이다. 이 아래로만 안 내려가면 된다:

| 패키지 | 최소 | 그 아래면 무슨 일이 나는가 |
|---|---|---|
| **pandas** | **2.0** | `pd.errors.ChainedAssignmentError` 가 없어 **무음 실패 탐지 장치가 조용히 꺼진다.** 예외가 나는 게 아니라 보호장치만 사라지는 거라 더 위험하다 |
| **numpy** | **1.17** | `np.random.default_rng` 없음 → 부트스트랩 신뢰구간(3장)·permutation 중요도(7장)가 즉시 죽는다 |
| scipy | 제약 없음 | `stats.beta.ppf`·`spearmanr` 만 쓴다. numpy 2.x 환경이면 scipy 1.13+ 가 필요하지만 그건 numpy 쪽 제약 |
| scikit-learn · lightgbm · statsmodels | 제약 없음 | 버전 민감 API 사용 0건 |
| tensorflow | Keras 3 (2.16+) | 포드의 2.17 로 충분. **절대 올리지 말 것**(8절) |

**KAMP 포드(pandas 2.1.4 · numpy 1.26.4)는 이 하한을 전부 만족한다.**

## 5. 실행 시간

- **FULL(기본)**: CPU 기준 45~75분. Optuna 50회 탐색 + DNN 300에폭 + 6.6절 변형 62개(약 13분)가 대부분이다.
- **빠른 확인**: `KAMP_FAST=1` → 탐색 3회·에폭 5회 (코드는 동일).
  **수치가 본실행 값이 아니므로 동작 확인용으로만** 쓸 것.
  FAST 로 먼저 돌렸다면 `mv outputs outputs_fast` 로 치워두고 FULL 을 돌린다.

## 6. 재현성 — 두 가지만 지키면 된다

1. **`PYTHONHASHSEED=42` 는 커널이 뜨기 전에 줘야 한다.**
   노트북 안의 `os.environ.setdefault(...)` 는 이미 실행 중인 인터프리터에 반영되지 않는다.
   `run.sh` 는 넣어서 실행한다. 주피터 UI 면 서버 자체를 `PYTHONHASHSEED=42 jupyter lab` 으로.
2. **셀은 위에서 아래로 한 번만, 순서대로.** 1부를 건너뛰고 2부를 돌리면 죽는다.

`TF_DETERMINISTIC_OPS=1`, `TF_ENABLE_ONEDNN_OPTS=0` 은 노트북 첫 셀이
TensorFlow 임포트보다 먼저 설정한다(순서가 보장되도록 조립돼 있다).

## 7. 한글 폰트

노트북은 이 순서로 폰트를 고른다:

1. 환경변수 `KAMP_FONT` 로 지정한 이름
2. 알려진 이름 (Malgun Gothic → NanumGothic → Noto Sans KR → AppleGothic)
3. **설치된 폰트 중 한글 글리프(`가`)가 실제로 있는 것** — 배포판마다 이름이
   달라서(`Noto Sans CJK KR` 등) 이름만으로는 못 잡는다. cmap 을 직접 확인한다.

셋 다 실패하면 경고를 내고 기본 폰트로 간다. **코드는 죽지 않고 그림의 한글만 깨진다(□□□).**

**KAMP 포드에는 NanumGothic 이 이미 있어서 폰트 문제는 없었다.**
`setup_pod.sh` 의 "root 가 아니라 apt 설치를 건너뛴다" 경고는 **정상이며 무시해도 된다.**

## 8. GPU

- GPU 를 쓰는 건 **DNN(TensorFlow) 부분뿐**이다.
  LightGBM · Optuna · RandomForest · SHAP 는 CPU 를 쓴다.
- **KAMP 포드: Tesla V100-32GB, TF 2.17.0 이 이미 GPU 를 잡고 있다.**
  → **TensorFlow 를 절대 업그레이드하지 말 것.** CUDA 연동이 깨질 수 있다.
- 확인: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- CPU 로 돌아도 결과는 거의 같고 그 부분만 느려진다(10절의 DNN 항목 참고).

## 9. 실제로 막혔던 것과 해결 (중요)

**증상** — 처음 `bash setup_pod.sh` 를 돌렸을 때:

```
[2/4] numpy · pandas 선행 설치
ERROR: Could not install packages due to an OSError: [Errno 13] Permission denied: 'METADATA'
```

**원인** — 초기 스크립트가 `requirements.txt` 의 핀(`numpy==2.3.5`, `pandas==3.0.5`)을
그대로 설치하려 했다. 포드는 공유 conda(`/opt/conda`)라, **새 패키지를 넣는 건 되지만
이미 있는 numpy 를 갈아끼우려다 권한에서 막힌 것**이다.
게다가 pip 가 교체 도중 죽으면서 `~umpy` 잔해가 남아 이후 명령마다 경고가 떴다.

**해결** — 공유 환경에서는 **핀을 강제하지 말고, 없는 것만, 기존 버전을 묶어서** 깐다:

```bash
# (1) numpy 가 멀쩡한지 확인하고 잔해 제거
python -c "import numpy,pandas; print(numpy.__version__, pandas.__version__)"
rm -rf /opt/conda/lib/python3.11/site-packages/~umpy

# (2) 기존 버전을 constraints 로 묶고 없는 것만 설치
python - <<'PY' > constraints.txt
import importlib.metadata as md
for p in ["numpy","pandas","scipy","scikit-learn","matplotlib","lightgbm",
          "statsmodels","tensorflow","keras"]:
    try: print(f"{p}=={md.version(p)}")
    except Exception: pass
PY
pip install --no-cache-dir -c constraints.txt optuna shap
```

**`-c constraints.txt` 가 핵심이다.** 실제로 이게 `shap` 을 0.51.0 대신 **0.49.1 로 스스로
내려가게** 만들어 numpy·TensorFlow 를 지켜냈다. 이게 없으면 shap 이 의존성 해석 중에
numpy 를 올리고 **GPU 연동이 조용히 깨진다.**

`setup_pod.sh` 는 이제 이 방식으로 동작한다(설치 전 constraints 자동 생성, `~xxx` 잔해 감지).
다만 **포드에서 실제로 검증된 것은 위 수동 명령**이다. 스크립트가 막히면 위를 직접 실행한다.

기타:
- `--user` 로 깔고 싶으면 `bash setup_pod.sh` 가 권한을 보고 알아서 전환한다.
- 홈에 격리하고 싶으면 `bash setup_pod.sh --venv` (기존 TF 는 `--system-site-packages` 로 재사용).
- `bash setup_pod.sh --pin` 은 **단독 환경 전용**이다. 공유 포드에서 쓰면 위 사고가 재현된다.

## 10. 검증 결과

**이번 버전(361셀) — 개발 PC 에서 FULL 실행 (2026-09-25)**

```
코드셀 272 · 실행됨 272 · 에러 0
표 106개 · 그림 39개 · 예측결과 있음 · 모델 번들 full/eval · full/deploy 있음
완주 확인 (약 44분)
```

- 예측결과 파일은 이전 버전과 **바이트 동일**, 개발 저장소의 전체 노트북과도 바이트 동일.
- 표 106개 중 개발 저장소 결과와 바이트가 다른 것은 **학습시간(벽시계) 열이 있는 3개**(`ch2_regression`·`ch2_scorecard`·`timing`)뿐.
- 모델 번들 ID `eval-full-7be5289680f8` · `deploy-full-d8293292a723` 가 개발 저장소와 같고, `serving verify` 자가검증 통과,
  `serving replay`(9/1~9/14) 결과가 예측결과 파일과 같다(MAE 5.2199).
- **포드에서 이번 버전을 FULL 로 다시 돌려야 한다.** 포드는 라이브러리 버전이 달라(lightgbm 4.5.0 등) 번들 ID·해시는 달라지는 게 정상이다.
  대신 `outputs/predictions_test_336h.csv` 와 `outputs/tables/ch2_lag_decomposition.csv` 가 이번 결과와 같아야 한다
  (이전 버전의 성능표는 버전이 달라도 같았다 — 아래).

**이전 버전(325셀) — KAMP 포드 실행 (2026-09-23, FAST)**

```
코드셀 238 · 실행됨 238 · 에러 0
표 100개 · 그림 39개 · 예측결과 있음
완주 확인
```

포드 환경: JupyterHub / 공유 conda `/opt/conda` / Python 3.11.9 / Tesla V100-32GB /
numpy 1.26.4 · pandas 2.1.4 · scipy 1.11.4 · scikit-learn 1.4.2 · matplotlib 3.9.2 ·
lightgbm 4.5.0 · statsmodels 0.14.2 · tensorflow 2.17.0 · optuna 5.0.0 · shap 0.49.1

**버전 차이가 결과에 미치는 영향** — 포드와 **동일한 11개 패키지 버전**으로 별도 환경을
만들어 돌린 뒤, 개발 환경(pandas 3.0.5 · TF 2.21)의 결과와 대조했다:

| | 결과 |
|---|---|
| 표 100개 | **90개 바이트 동일** |
| 예측결과 파일 | **바이트 동일** |
| 최종 선정 모델 | **동일** (`2단계 레짐(3분류)`, 종합점수 0.997) |
| 다른 표 10개 | 버전문자열·실행시간 4개 / 1e-6 부동소수점 2개 / 동점 정렬순서 2개 / **DNN 관련 3개** |

다른 표 중 **모델 성능이 실제로 다른 것은 `DNN (MLP)` 한 줄뿐**이다(τ 156.7 → 148.1 등).
원인은 TensorFlow 버전 차이이고, **신경망은 버전·하드웨어 간 비트 재현이 되지 않는다.**
DNN 은 비교용 베이스라인이고 종합점수 최하위(0.167)라
**최종모델 선정과 제출 예측파일에는 영향이 없다.**

> 포드는 GPU 라 DNN 수치가 또 조금 달라질 수 있다.
> **보고서에 DNN 수치를 인용했다면 포드 실행값으로 다시 맞춰야 한다.**

**알아둘 것 — 동점일 때 순서가 버전마다 바뀐다**

`ch1_profile_dup.csv` 의 **그룹 번호**가 버전에 따라 달라진다. 내용(45개 그룹, 날짜목록,
잉여일 115)은 완전히 같고 **번호와 행 순서만** 다르다. 원인은 두 줄이다:

- `s01_diagnose.py:186` `hashes.value_counts()` — 동점(같은 개수) 그룹의 순서가 버전마다 다름
- `s01_diagnose.py:203` `sort_values(...)` — 기본 정렬이 **불안정(quicksort)** 이라 동점 순서를 보존하지 않음

`kind="mergesort"` 를 주면 세 버전(1.5.3 / 2.2.3 / 3.0.5)이 모두 같은 순서가 되는 것까지 확인했다.
**지금은 고치지 않았다** — 고치면 기존 보고서에 실린 그룹 번호와 어긋날 수 있어서다.

> ⚠️ 같은 불안정 정렬이 **`s06_eval.py:208` 의 최종모델 선정**(`sort_values("OOF MAE")` → `idxmin()`)
> 에도 쓰인다. **OOF MAE 가 동점인 모델이 둘 이상이면 최종 제출 모델이 버전에 따라 바뀔 수 있다.**
> 이번 실행에서는 동점이 아니었고(1위 0.997 · 2위 0.966) 양쪽 환경에서 같은 모델이 뽑혔다.

**아직 검증 안 한 것**: 포드에서의 **FULL 실행**(위 검증은 전부 `KAMP_FAST=1`),
그리고 `setup_pod.sh` 자동 설치 경로(포드에서는 9절의 수동 명령으로 설치했다).

## 11. 막혔을 때

| 증상 | 원인 / 해결 |
|---|---|
| `Permission denied: 'METADATA'` | 9절 — 핀 강제가 원인. constraints 방식으로 |
| `Ignoring invalid distribution ~umpy` | 9절 — 이전 설치 실패 잔해. 정상 임포트 확인 후 `rm -rf` |
| `root 가 아니라 apt 설치를 건너뛴다` | 정상. 포드엔 NanumGothic 이 이미 있다(7절) |
| `NameError: name 'df' is not defined` | 셀을 건너뛰고 실행했다. 1부부터 순서대로 다시 |
| 그림 한글이 □□□ | 폰트 문제(7절). 폰트 넣은 뒤 **커널 재시작** 필요 |
| `ModuleNotFoundError` | `bash check_env.sh` 로 무엇이 없는지 확인 후 9절 |
| 30분짜리 실행이 세션 끊겨 죽음 | `nohup bash run.sh > full_run.out 2>&1 &` |
| 커널이 계속 죽는다 | 메모리 부족일 수 있다. `KAMP_FAST=1` 로 먼저 확인 |

## 12. (선택) 저장된 모델로 예측 — `serving/`

노트북을 끝까지 돌리면 `outputs/models/full/{eval,deploy}/` 에 모델 번들이 생긴다.
`serving/` 은 이 번들을 **해시로 검증하며 읽어** 익일 24시간을 예측하는 추론 전용 코드다(학습 코드는 없다).
배치 명령은 **lightgbm · numpy · pandas 만** 쓰므로 포드에 이미 있는 패키지로 돈다
(포드와 같은 pandas 2.1.4 · numpy 1.26.4 · lightgbm 4.5.0 환경에서 단위 테스트 76개 통과).

```bash
# 번들 무결성·자가검증 (파일 해시 · 버전 · 저장된 기준 사례 재현)
python -X utf8 -m serving verify --bundle outputs/models/full/deploy

# 평가 번들로 9/1~9/14 를 하루씩 재생 — 결과가 outputs/predictions_test_336h.csv 와 같아야 한다
python -X utf8 -m serving replay --bundle outputs/models/full/eval        --data data/okm_augumented_2021.csv --start 2021-09-01 --end 2021-09-14 --out outputs/replay_eval.csv

# 익일 예측 — 이력 CSV(최근 8~62일, 권장 14일) + 대상일 생산계획 24행
python -X utf8 -m serving predict --bundle outputs/models/full/deploy        --history history.csv --plan plan.csv --out pred.csv
```

- 번들은 **만든 환경과 같은 lightgbm·numpy·pandas 버전**에서만 열린다(다르면 `VERSION_MISMATCH`). 포드에서 만든 번들은 포드에서 쓴다.
- 계획 CSV 에 전력 컬럼이 있으면 거부한다(미래 정보 차단). 공휴일은 `serving/config/holidays_kr.json` 을 쓴다.
- REST API(`serving/api.py`)는 fastapi·uvicorn 이 필요하다. 포드에는 없으므로 쓰려면 9절 방식(constraints)으로 설치한 뒤
  `KAMP_BUNDLE_DIR=outputs/models/full/deploy python -m uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port 8000`.
- 입력 형식·오류 코드는 `serving/README.md`. 그 문서의 `bash run.sh ...`·`make ...` 명령은 개발 저장소용이다(이 패키지의 `run.sh` 에는 없다).
