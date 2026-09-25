# 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석

**제6회 K-인공지능 제조데이터 분석 경진대회 — 과제 ⑤ 자원 최적화**

선박엔진용 볼트·너트 제조공장의 2021년 1~9월 전력·생산·기상 데이터(6,168행)로
① 다음 날 시간별 전력사용량 예측과 ② 최대수요 피크 위험 사전 탐지를 수행합니다.

---

## 30초 요약

| | |
|---|---|
| **최종모델** | 2단계 레짐 모델 (가동상태 3분류 후 상태별 회귀) |
| **예측 성능** | 테스트 MAE **5.220** — 최선 베이스라인(6.039) 대비 **13.56% 개선** |
| **피크 탐지** | Recall **0.929** / F1 **0.525** |
| **피크 저감** | 최대수요 **13.2 kW** 감소 (연간 기본요금 약 130만원) |
| **추론 속도** | 336시간 배치 **7.43초** — 매일 반복 실행 가능 (모델 재적합을 포함한 측정치이며 벽시계라 실행마다 3~8초로 변동 · 이번 값은 다른 FULL 실행과 CPU 를 나눠 쓴 상태 — 저장된 번들로 예측만 하면 번들 순수 추론 16.3ms, 로컬 API 1회 실측 약 64ms/요청) |
| **검증** | 테스트 427개 통과 · 재현성 22개 표 1,988개 셀 소수 6자리 일치 (모델 번들 바이트 포함, 같은 OS 기준) |

정직하게 밝혀 둘 것: **MAE 개선은 통계적으로 유의하지 않습니다**(p=0.067).
이유와 대응은 [`03_RESULTS.md`](03_RESULTS.md) 에 적어 두었습니다.

---

## 3단계 실행

```bash
# 1) 셋업 (한글 폰트 + 패키지 + Jupyter 커널)   — 3~5분
bash setup_pod.sh

# 2) 환경 점검 — 지금 이 환경에서 돌아갈지 판정  — 30초
#    Jupyter 에서 START_HERE.ipynb 열고 Run All
#    또는:
bash run.sh check

# 3) 메인 노트북 실행                          — CPU 30~60분
bash run.sh run
```

`make` 가 있는 환경이라면 `make check` / `make run` 도 동일하게 동작합니다.
`make` 가 없는 pod 도 있어 **`run.sh` 를 기본 경로로** 둡니다.

자세한 pod 명령은 [`01_QUICKSTART_POD.md`](01_QUICKSTART_POD.md) 를 보세요.

> ### ⚠️ GPU 는 필요하지 않습니다
>
> 주력 모델 **LightGBM 은 CPU 전용** 빌드이고, TensorFlow 모델은 SimpleRNN(50유닛)·
> MLP(128-64)로 학습 표본이 약 5,600행뿐입니다. GPU 로 보내는 오버헤드가 이득보다 큽니다.
>
> GPU pod 에서도 문제없이 돌아가지만 **CPU pod 이 더 저렴합니다.**
> 이 사실을 숨기면 나중에 "왜 GPU 를 써도 안 빨라지나" 하는 오해가 생기므로 미리 적습니다.

---

## 폴더 지도

```
00_README.md              ← 지금 읽는 파일
01_QUICKSTART_POD.md      pod 에 복붙할 명령만
02_PIPELINE_GUIDE.md      코드가 무엇을 왜 하는지 (설계 근거 포함)
03_RESULTS.md             결과 수치 + 정직한 한계 4가지
04_TROUBLESHOOTING.md     안 될 때 보는 문서

START_HERE.ipynb          ★ pod 에서 제일 먼저 실행 (환경 점검)
자원최적화_제안모델.ipynb   ★ 메인 산출물 164셀 (이미 실행된 상태로 동봉)

setup_pod.sh              원샷 셋업 (여러 번 실행해도 안전)
run.sh                    실행 래퍼 (make 없는 환경용) — check/run/test/finalize/verify-head/bundle-verify/serve 등 (`bash run.sh` 로 목록)
Makefile                  make 가 있으면 동일 기능
requirements.txt          런타임 의존성 (완전 핀)
requirements-dev.txt      재빌드·테스트용 추가

data/                     학습 데이터 (6,168행 × 18열, 428KB)
outputs/                  ★ 우리가 돌린 결과 — 기대값 대조용
  figures/                  그림 39장 + figure_index.csv (그림↔보고서 절 매핑)
  tables/                   표 114개 (ch6_model_bundles = 모델 번들 파일별 sha256, ch2_lag_* = 6.6절 지연변수 제거 효과 분해)
  models/                   ★ 모델 번들 (노트북 10.5절이 생성 · git·제출 zip 제외)
    full/eval/                제출 예측을 만든 모델 (학습 ~08-31)
    full/deploy/              배포용 재학습 (학습 ~09-14)
  predictions_test_336h.csv  제출용 예측결과 (336행)
  report_tbd_filled.md      보고서 빈칸 채움표 + 문장 치환사전
  report_ch4~6_draft.md     보고서 4·5·6장 본문 초안
  발표자료.pptx              발표자료 골격 9슬라이드

src/                      노트북 소스 s00~s10 (★ 단일 진실원)
tools/                    빌드·마감·재현성 검사 스크립트
                            build_notebook.py      src/ → ipynb 조립
                            build_serving_core.py  src/ → serving/_core.py 생성 (서빙 코드 한 벌)
                            build_experiment_notebook.py  src/ → 포드 제출용 단일 노트북(자원최적화_실험전용.ipynb) 조립
                            build_mindmap_canvas.py  위 생성기가 쓰는 절 파서 (코드 지도 생성)
                            finalize_notebook.py   실행 후 마감 (필수)
                            check_reproducibility.py  동일 시드 2회 비교 / --baseline-ref HEAD
                            start_here_src.py      START_HERE.ipynb 의 소스
serving/                  모델 번들 서빙 — 배치 CLI + REST API (제출 zip 포함, 번들은 노트북 실행으로 생성)
                            README.md              운영 런북 (입력 계약·오류 코드·배포)
                            requirements.txt       서빙 전용 의존성 (fastapi 등)
tests/                    검증 테스트 427개 (서빙·번들 181개 포함)

reference/                대회 원본 자료 (과제공개, 보고서양식, 가이드북 원본)
보고서/                    결과보고서 v3 (docx · 미리보기 PDF) — ①보고서 제출물의 작업본. 서명·설문 캡처 전이라 최종 PDF 아님
video/                    해설·강의 영상 (1080p, 한글 자막 트랙 내장) — 목차는 video/README.md
  ep01~ep06_*.mp4           개요 해설 6편 (43분)
  모델/ · 평가지표/          모델 9편 · 평가지표 13편 강의 (2시간 53분)
  서빙/                     모델이 서비스가 되기까지 — 번들 저장·검증·예측·CLI/API 7편 (63분, 비개발자용 용어 정의 포함)
                            용량이 커서(합계 약 950MB) 제출 zip 에 넣을지는 따로 정하세요
experiments/              격리된 실험 기록 — 피처 제거 확인·게이트 수정안·보고서 docx 편집·포드 패키지 (제출 zip 제외, 색인 experiments/README.md)
_dev_docs/                개발 문서 — 제출 zip 에는 넣지 마세요
```

### 두 가지만 기억하면 됩니다

1. **`data/` 와 `outputs/` 이름은 바꾸지 마세요.** 노트북이 상대경로로 찾습니다.
   그리고 항상 **이 폴더를 작업 디렉터리로** 삼으세요.

2. **`src/` 가 원본이고 노트북은 빌드 산출물입니다.** 코드를 고칠 일이 있으면
   `src/sNN_*.py` 를 고친 뒤 `bash run.sh build`(make 가 있으면 `make build`)로 노트북을 다시 만듭니다.
   노트북 셀을 직접 고치면 다음 빌드에서 덮어써집니다.

---

## 학습된 모델 — 서비스 번들

노트북을 실행하면 10.5절이 모델을 **파일로 저장**하고, 그 자리에서 다시 읽어 검증합니다.

| 번들 | 학습 구간 | 쓰임 |
|---|---|---|
| `outputs/models/full/eval/` | ~ 2021-08-31 | 제출 예측을 만든 바로 그 모델 — 다시 읽은 예측이 제출 파일과 **비트 단위로 같다** |
| `outputs/models/full/deploy/` | ~ 2021-09-14 | **배포용** 재학습 (보고서 수치에는 쓰지 않음) |

구성: 최종모델(2단계 레짐 3분류) · 피크 직접분류기 · 확률 보정기 · 예측구간 · 임계값(τ·θ) · 피처 계약.
형식은 LightGBM 네이티브 텍스트 + JSON 매니페스트(파일별 sha256)이며 pickle 은 쓰지 않습니다.

```bash
# (번들이 없으면) bash run.sh run   # FULL 30~60분 — outputs/models/full/ 생성
pip install -r serving/requirements.txt
bash run.sh bundle-verify                       # 번들 무결성·자가검증
python -X utf8 -m serving predict --bundle outputs/models/full/deploy \
       --history history.csv --plan plan.csv --out pred.csv     # 익일 24시간 예측
bash run.sh serve                               # REST API (127.0.0.1:8000)
```

입력 계약·오류 코드·배포 절차는 [`serving/README.md`](serving/README.md) 를 보세요.
`serving/` 은 제출 zip 에 포함하고, `outputs/models/` 는 **git·제출 zip 모두 제외**입니다(용량 약 51MB).
새로 받은 환경에서는 먼저 `bash run.sh run` 으로 번들을 만든 뒤 위 명령을 쓰세요.

---

## 무엇이 들어 있나

`outputs/` 에는 **이미 실행이 끝난 결과**가 들어 있습니다. 직접 돌려 보기 전에
결과만 먼저 보고 싶다면 이 순서를 권합니다.

| 보고 싶은 것 | 파일 |
|---|---|
| 결론과 수치 | [`03_RESULTS.md`](03_RESULTS.md) |
| 보고서에 넣을 값 전부 | `outputs/report_tbd_filled.md` |
| 그림 (보고서 어느 절에 쓰는지 포함) | `outputs/figures/figure_index.csv` |
| 모델 성능 비교표 | `outputs/tables/ch2_regression.csv`, `ch2_peak_detection.csv` |
| 피크 저감 시나리오 | `outputs/tables/ch4_scenarios.csv` |
| 4·5·6장 본문 초안 | `outputs/report_ch4_draft.md` · `ch5` · `ch6` |
| 영상으로 이해하기 (개요 6편 + 모델·평가지표 강의 22편 + 서빙 e2e 7편) | [`video/README.md`](video/README.md) |
| 결과보고서 (현재 작업본 v3) | `보고서/` (docx · 미리보기 PDF) |
| 제출 뒤 분석·서비스 제안 (피처 제거 확인, 휴무 오분류 수정안) | [`experiments/README.md`](experiments/README.md) (저장소에만 있음 · 제출 zip 미포함) |

---

## 제출물과의 관계

경진대회 제출물은 4종입니다. 이 패키지가 그중 **②소스코드**에 해당합니다.

| 제출물 | 이 패키지에서 |
|---|---|
| ① 보고서 PDF | `outputs/report_tbd_filled.md` 의 채움표·치환사전으로 작성 |
| ② **소스코드 zip** | **이 폴더 전체** — 단 `_dev_docs/`·`experiments/`·`outputs/models/`·각 폴더의 `CLAUDE.local.md`·`CLAUDE.md`(개발 안내, 제외 권장)·`.git/`·`.omc/`·`.cache/`·모든 `__pycache__/`·`outputs/*.log`·빈 `outputs/baseline_repro/` 는 제외 (`video/` 는 용량 때문에 별도 결정). 로그·바이트코드에는 설치 경로(계정명)가 박히는데 개인정보 스캔 대상이 아니다 |
| ③ 발표자료 | `outputs/발표자료.pptx` 골격을 다듬어 사용 |
| ④ 설문 캡처 | 사람이 직접 (`outputs/report_tbd_filled.md` §6 참조) |

**블라인드 평가**이므로 소속·학교·로고를 넣지 마세요(성명·팀명만 허용).
`bash run.sh finalize` 가 노트북에서 개인정보를 후스캔합니다.
`_dev_docs/` 에는 개발 이력이 들어 있어 제출 대상이 아닙니다.
위 제외 목록은 git 추적 여부와 별개입니다 — **작업 폴더를 그대로 압축하지 말고 목록대로 빼세요.**

**포드 제출용 단일 노트북 패키지**(보고서 6장이 기술하는 구성: 노트북 1개 + 데이터 + 실행 스크립트 + `serving/`)는
`python -X utf8 experiments/2026-09-25_pod_package/make_pod_package.py --out <패키지 폴더>` 로 폴더와 zip 을 한 번에 만듭니다.
그 안의 노트북은 `tools/build_experiment_notebook.py` 가 같은 `src/` 에서 조립합니다(노트북만 따로 만들 때는 `--out <폴더>/자원최적화_실험전용.ipynb`, 폴더는 미리 만들어 둘 것).
보고서·제출물 생성 절(6.7·8.5·9장·10.2~10.4·10.6·최종 게이트)만 빠지고 **모델 저장(10.5)과 6.6절은 포함**됩니다.
