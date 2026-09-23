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
| **추론 속도** | 336시간 배치 **4.91초** — 매일 반복 실행 가능 |
| **검증** | 테스트 236개 통과 · 재현성 722개 셀 소수 6자리 일치 |

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
03_RESULTS.md             결과 수치 + 정직한 한계 3가지
04_TROUBLESHOOTING.md     안 될 때 보는 문서

START_HERE.ipynb          ★ pod 에서 제일 먼저 실행 (환경 점검)
자원최적화_제안모델.ipynb   ★ 메인 산출물 159셀 (이미 실행된 상태로 동봉)

setup_pod.sh              원샷 셋업 (여러 번 실행해도 안전)
run.sh                    실행 래퍼 (make 없는 환경용) — check/run/test/finalize
Makefile                  make 가 있으면 동일 기능
requirements.txt          런타임 의존성 (완전 핀)
requirements-dev.txt      재빌드·테스트용 추가

data/                     학습 데이터 (6,168행 × 18열, 428KB)
outputs/                  ★ 우리가 돌린 결과 — 기대값 대조용
  figures/                  그림 39장 + figure_index.csv (그림↔보고서 절 매핑)
  tables/                   표 108개
  predictions_test_336h.csv  제출용 예측결과 (336행)
  report_tbd_filled.md      보고서 빈칸 채움표 + 문장 치환사전
  report_ch4~6_draft.md     보고서 4·5·6장 본문 초안
  발표자료.pptx              발표자료 골격 9슬라이드

src/                      노트북 소스 s00~s10 (★ 단일 진실원)
tools/                    빌드·마감·재현성 검사 스크립트
                            build_notebook.py      src/ → ipynb 조립
                            finalize_notebook.py   실행 후 마감 (필수)
                            check_reproducibility.py  동일 시드 2회 비교
                            start_here_src.py      START_HERE.ipynb 의 소스
tests/                    검증 테스트 236개

reference/                대회 원본 자료 (과제공개, 보고서양식, 가이드북 원본)
video/                    해설·강의 영상 (1080p, 한글 자막 트랙 내장) — 목차는 video/README.md
  ep01~ep06_*.mp4           개요 해설 6편 (43분)
  모델/ · 평가지표/          모델 9편 · 평가지표 13편 강의 (2시간 53분)
                            용량이 커서(합계 약 700MB) 제출 zip 에 넣을지는 따로 정하세요
_dev_docs/                개발 문서 — 제출 zip 에는 넣지 마세요
```

### 두 가지만 기억하면 됩니다

1. **`data/` 와 `outputs/` 이름은 바꾸지 마세요.** 노트북이 상대경로로 찾습니다.
   그리고 항상 **이 폴더를 작업 디렉터리로** 삼으세요.

2. **`src/` 가 원본이고 노트북은 빌드 산출물입니다.** 코드를 고칠 일이 있으면
   `src/sNN_*.py` 를 고친 뒤 `make build` 로 노트북을 다시 만듭니다.
   노트북 셀을 직접 고치면 다음 빌드에서 덮어써집니다.

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
| 영상으로 이해하기 (개요 6편 + 모델·평가지표 강의 22편) | [`video/README.md`](video/README.md) |

---

## 제출물과의 관계

경진대회 제출물은 4종입니다. 이 패키지가 그중 **②소스코드**에 해당합니다.

| 제출물 | 이 패키지에서 |
|---|---|
| ① 보고서 PDF | `outputs/report_tbd_filled.md` 의 채움표·치환사전으로 작성 |
| ② **소스코드 zip** | **이 폴더 전체** (단 `_dev_docs/` 는 제외) |
| ③ 발표자료 | `outputs/발표자료.pptx` 골격을 다듬어 사용 |
| ④ 설문 캡처 | 사람이 직접 (`outputs/report_tbd_filled.md` §6 참조) |

**블라인드 평가**이므로 소속·학교·로고를 넣지 마세요(성명·팀명만 허용).
`bash run.sh finalize` 가 노트북에서 개인정보를 후스캔합니다.
`_dev_docs/` 에는 개발 이력이 들어 있어 제출 대상이 아닙니다.
