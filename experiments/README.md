# experiments/ — 격리된 실험 기록

제출 노트북·보고서 수치와 **분리된** 분석·도구 기록이다. 이 폴더의 어떤 것도 노트북 빌드(`src/`만 읽음),
테스트(`pytest.ini` 의 `testpaths = tests`), 재현성 검사(`outputs/` 만 비교), 마감 스캔(노트북·`outputs/`)에 들어가지 않는다.
**제출 zip 에는 넣지 않는다.**

| 폴더 | 무엇 | 결론 | 제출물 영향 |
|---|---|---|---|
| [`2026-09-25_feature_ablation/`](2026-09-25_feature_ablation/) | Ablation 에서 "지연변수를 빼면 좋아진다" 는 결과가 최종모델을 바꿀 근거인지 검증. 변형 62개 + D2 | **최종모델 유지.** 개선은 8/2·8/3 이틀의 게이트 오분류가 사라진 착시였다 | 노트북 **6.6절**로 이식됨(같은 수치 재현) |
| [`2026-09-25_gate_fix/`](2026-09-25_gate_fix/) | 그 게이트 오분류를 고치는 수정안(R·G·RG, 사후 D·R-soft) 비교 + 서빙 시제품 | **사전 기준 채택 없음.** 계획 휴무를 믿게 하는 수정은 ERP 결측 가동일에서 실패한다. 제안: 확정 휴무일에만 거는 서빙 옵션(기본 꺼짐) → `PROPOSAL.md` | 없음 (서비스 제안용) |
| [`2026-09-25_report_edits/`](2026-09-25_report_edits/) | 결과보고서 docx v2·v3 를 만든 편집 스크립트 | 원본 docx 는 그대로, 새 파일로 저장 | 보고서 파일(저장소 밖) |
| [`2026-09-25_pod_package/`](2026-09-25_pod_package/) | 포드 제출용 단일 노트북 패키지 조립 스크립트와 패키지 전용 파일 | 361셀 노트북 + data + 스크립트 + `serving/` | 제출 패키지를 만든다 |

## 격리 규칙

- **저장소 `outputs/` 에 쓰지 않는다.**
  - 실험 스크립트는 파이프라인(`src/s00~s03`)을 import 하는데, import 하는 순간 작업 디렉터리의 `outputs/` 에 표·그림을 쓴다.
  - 그래서 각 실험 폴더의 `_work/`(데이터 사본만 둠, git 제외)로 이동한 뒤 import 한다.
  - 한때 사본 저장소에서 돌려 그림 색인이 덮어써진 적이 있다.
- **보관된 결과를 덮지 않고 재실행하려면** 환경변수로 출력 위치를 바꾼다.
  - `KAMP_EXP_OUT`: 결과 CSV
  - `KAMP_EXP_FIG`: 그림
  - 기본값은 각 폴더의 `results/`·`figures/` 다. 상대경로를 주면 실행한 위치 기준으로 풀린다(스크립트가 chdir 전에 절대경로로 고정).
- 판정은 교차검증(OOF)으로만 한다. 테스트 구간은 판정 뒤에 보고만 한다. 각 실험의 `PLAN.md` 가 실행 전에 기준을 고정했다.
- 스크립트·문서에 절대경로·계정명을 쓰지 않는다. 개인 경로가 필요한 것(보고서 폴더)은 환경변수(`KAMP_REPORT_DIR`)로 받는다.

## 재실행 (저장소 루트에서, `PYTHONHASHSEED=42 KAMP_FAST=0`)

```bash
# 피처 제거 실험 (약 13분)
python -X utf8 experiments/2026-09-25_feature_ablation/run_feature_experiment.py
python -X utf8 experiments/2026-09-25_feature_ablation/check_gate_usage.py     # 게이트 분기·배정 점검

# 게이트 수정안 (순서대로, 약 10분) — 뒤 단계가 앞 단계 결과를 읽는다
G=experiments/2026-09-25_gate_fix
python -X utf8 $G/run_gate_fix.py            # 사전등록 1차 판정
python -X utf8 $G/posthoc_gate_fix.py        # 사후 진단
python -X utf8 $G/posthoc_datafix.py         # 가설 D
python -X utf8 $G/posthoc_holiday_soft.py    # 공휴일·약한 보정
python -X utf8 $G/serving_plan_override.py   # 서빙 시제품 (outputs/models/full/eval 필요 — FULL 실행 후)
python -X utf8 $G/make_figures.py            # 그림

# 포드 패키지
python -X utf8 experiments/2026-09-25_pod_package/make_pod_package.py --out <만들 폴더>
```

- 2026-09-25 에 새 위치에서 다시 돌려(`KAMP_EXP_OUT` 로 다른 곳에), 보관된 결과와 비교했다.
  - 결과 CSV 32개는 수치 차이 1e-9 이내(실행시간 열 제외)였다.
  - 서빙 로그와 그림 3장은 바이트 단위로 같았다.
- 각 실험의 `_original/` 은 바탕화면에서 실행할 때의 **손대지 않은** 계획서·주 스크립트다(사전등록 sha256 확인용, 이 위치에서는 실행되지 않는다).
- `run_gate_fix.py` 는 1차 당시 그림을 `figures/round1_*.png` 로 쓴다. 최종 그림(`fig1~3`)은 `make_figures.py` 가 그린다.
  - `posthoc_gate_fix.py` 도 `fig3_erp_missing_downside.png` 를 쓴다. 사후 스크립트만 다시 돌릴 때는 `KAMP_EXP_FIG` 를 주거나 `make_figures.py` 까지 돌린다.
- 결정적 LightGBM 설정이라 같은 OS·라이브러리에서는 같은 값이 나온다.
