# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

제6회 K-인공지능 제조데이터 분석 경진대회 과제 ⑤(자원 최적화) 제출 패키지다. 한 공장의 2021-01~09 시간별 데이터(`data/okm_augumented_2021.csv`, 6,168행)로
① 익일 24시간 전력(y_avg=평균, y_peak=peak15) 예측과 ② 피크(y_peak ≥ θ=187) 위험 탐지를 한다. 문서·코드 주석은 한국어다.

> ⚠️ 이 파일도 **블라인드 평가 스캔 대상**이다. `s10` 개인정보 스캔이 루트의 `*.md` 를 전부 제출 대상으로 읽는다.
> `outputs/CLAUDE.local.md` 도 s10 스캔(`outputs/*.md`)과 `finalize` 후스캔(`outputs/` 전체)에 들어간다.
> 여기와 모든 코드·주석·마크다운에 절대경로·계정명·이메일을 쓰지 말 것. pod 계정명이 소문자 영단어(그 관리자 계정명)일 수 있어
> 그 단어가 부분일치로 걸리므로 새 텍스트에는 쓰지 않는다.

## 명령

작업 디렉터리는 항상 이 폴더다(노트북·코드가 `data/`·`outputs/` 를 상대경로로 찾는다). Windows 에는 `make` 가 없으므로 `bash run.sh <target>` 을 쓴다.
공통 환경: `PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8`, 파이썬은 `python -X utf8`.

```bash
bash run.sh build            # src/ → 노트북 재조립 + serving/_core.py 재생성
bash run.sh test             # 전체 테스트 (KAMP_FAST=1, -m "not slow"). 427개 통과가 정상
bash run.sh run              # 노트북 FULL 실행(CPU 30~60분) + finalize
bash run.sh run-fast         # 축소 실행(수치는 최종값 아님)
bash run.sh finalize         # 실행 후 필수: stderr 제거·개인정보 후스캔·최종 게이트 확인
bash run.sh verify-head      # 방금 실행 결과 vs 커밋된 결과(재실행 없음) — "수치 변화 0" 증명
bash run.sh verify           # 동일 시드 2회 실행 비교(FAST)
bash run.sh serving-test     # 서빙 단위·코드생성 테스트만 (파이프라인 없음, 수 초)
bash run.sh bundle-verify    # 모델 번들 무결성·자가검증
bash run.sh replay           # 평가 번들로 9/1~9/14 재생 → .cache/replay_eval.csv (제출 예측과 비트 동일해야 함)
bash run.sh serve            # REST API (127.0.0.1, PORT 기본 8000, KAMP_BUNDLE_DIR 기본 outputs/models/full/deploy)
bash run.sh serving-test-pipeline   # 번들·리플레이·API 테스트 ⚠️ outputs/ 를 FAST 로 덮어씀
python -X utf8 tools/build_serving_core.py --check   # serving/_core.py 가 src 와 동기인지
python -X utf8 tools/build_experiment_notebook.py --out <폴더>/자원최적화_실험전용.ipynb   # 포드 제출용 단일 노트북만 조립(폴더는 미리 만들 것)
python -X utf8 experiments/2026-09-25_pod_package/make_pod_package.py --out <폴더>   # 포드 패키지(노트북·data·스크립트·serving) 폴더 + zip

# 단일 테스트
PYTHONHASHSEED=42 KAMP_FAST=1 python -X utf8 -m pytest tests/test_s05_models.py::test_tau_is_median_of_fold_taus -q
```

- 테스트는 `tests/conftest.py` 의 fixture(`s00`~`s10`)가 해당 stage 모듈을 **import 하는 순간 그 장까지 파이프라인 전체를 실행**한다(FAST 로 약 3분). `@pytest.mark.slow` 는 기본 제외.
- ⚠️ 파이프라인을 타는 테스트(`run.sh test`, `serving-test-pipeline`, `verify`)는 **커밋된 FULL 산출물 `outputs/` 를 FAST 결과로 덮어쓴다**. 끝나면 `git checkout -- outputs/` 로 되돌리거나 FULL 을 다시 돌린다. 모델 번들만 `outputs/models/fast/` 로 분리돼 있다.
- 서빙 의존성은 `serving/requirements.txt`(루트 `requirements.txt` 는 제출용 완전 핀이라 건드리지 않는다. 테스트가 그 내용을 검사한다).

## 구조 — 여러 파일을 읽어야 보이는 것

**`src/sNN_*.py` 가 단일 진실원, 노트북은 빌드 산출물.** jupytext percent 포맷(`# %%` 코드셀, `# %% [markdown]`). `tools/build_notebook.py` 가
`src/s[0-9][0-9]_*.py` 를 순서대로 이어붙여 자기완결 노트북 `자원최적화_제안모델.ipynb`(164셀)를 만든다. `# %% tags=["nb-strip"]` 셀(stage 간 import)은 조립 때 삭제되고,
노트북은 어떤 모듈도 import 하지 않아야 한다(`validate()` 가 `from sNN import`·`import serving`·LightGBM 파일 API·윈도 사용자 경로를 stray 로 거부).
노트북 셀을 직접 고치면 다음 빌드에서 사라진다. 커밋된 노트북·`outputs/` 는 FULL 실행 결과이므로 src 를 바꾸면 build → FULL run → finalize → verify-head 까지 해야 한다.

**stage 모듈은 import 부작용이 곧 실행이다.** 각 파일 최상위 코드가 그 장을 실행하고 앞 장을 nb-strip import 로 끌어온다(s05 는 4장이 고립되지 않도록 s04 를 일부러 import).
그래서 다른 코드가 src 의 순수 함수만 가져다 쓸 수 없다 → 서빙은 코드 생성으로 해결한다(아래).

장별 역할과 설계 근거는 `02_PIPELINE_GUIDE.md` 에 있다. 코드만 봐서는 알 수 없는 계약:

- **6.6절(변수 제거 결과 검증)** 이 6.3 Ablation 의 "지연변수를 빼면 OOF MAE 가 좋아진다" 를 사전 기준으로 판정해 **44개 피처 유지**를 결론낸다.
  개선은 8/2·8/3(하계휴가 중 평일)에 게이트가 계획 휴무일을 가동으로 보낸 오분류가 사라진 착시다. 피처를 빼서 OOF 를 줄이려 하지 말 것.
  `get_fold_data`·`run_cv` 의 `frame=` 인자(기본 `feat`)는 이 절 전용이다 — 전역 `feat` 를 바꿔치기하지 말 것. 옛 6.6(보고서 서술 교정)은 6.7 이다.

- **peak15 = max(15분,30분,45분,60분)** (15분 단일 컬럼이 아니다). `평균 = floor(4구간 평균 + 0.5)` half-up — numpy round(은행가 반올림)를 쓰면 무결성이 깨진다.
- **θ(피크 정의)=187 고정 vs τ(경보 임계)** — θ 는 학습구간 q95 1회 산출(게이트 1이 187.0 을 강제)이고 피처 `roll*_peak_cnt` 에도 쓰인다. τ 는 fold 검증구간마다 F1 최대로 뽑고 그 중앙값을 쓴다.
- **예측 원점 = D일 24:00.** 안전 lag 는 24/48/168 뿐, 이동통계는 D일 23시 행에서 1회 계산해 D+1 24시간에 브로드캐스트한다(`rolling().shift(1)` 금지). 2장 게이트가 "대상일 전력을 지워도 피처가 같은가"(미래맹검)로 누수를 기계 검증한다.
- **fold1 폐기(4-fold), 데이터 조건 D1/D2**(D2 = 합성 복제일 제거). 마스크 `is_warmup`·`is_outage`·`is_erp_missing` 는 학습·평가에서 제외, 테스트 336행은 전부 유지.
- 가동 캘린더는 전력이 아니라 **`일생산량 == 0`** 으로 판정한다(전력 채널이 증강으로 덮어씌워진 날이 있다).
- 각 장 끝의 게이트 1~4는 실패 시 `AssertionError`. 6장 게이트(`gate5_eval.csv`, 03_RESULTS 의 "게이트 5")는 표로만 남는다 — MAE 유의성 1항목은 의도적으로 False 다.
  최종 게이트(`gate6_final.csv`, 15항목)도 노트북에서는 표로만 남고, `finalize` 는 **이것만** 강제한다.

**모델 번들 (s10 10.5절) → 서빙 (`serving/`)**

- s05 의 fit 함수가 적합 객체를 `_artifacts` 키로 돌려주고 `run_test` 가 전달한다. **키 이름 `_models` 를 쓰지 말 것** — s06 `measure_inference_time` 이 그 키로 분기해 보고서 INFER_SEC 가 바뀐다(INFER_SEC 는 현재 재적합 시간을 잰다, 알려진 결함).
- 10.5절이 `outputs/models/{full|fast}/{eval|deploy}/` 를 쓴다: eval = 제출 예측을 만든 객체 그대로(~08-31), deploy = 같은 레시피로 ~09-14 재학습(τ·보정기·구간 반폭은 eval 에서 상속).
  형식은 LightGBM 텍스트(`.lgb`) + 결정적 JSON 매니페스트(파일별 sha256 + `manifest_sha256`, `bundle_id` 끝 12자리). 저장 직후 다시 읽어 비트 일치·재적합 sha 재현을 검증하고 어긋나면 멈춘다.
- **pickle 금지**(레짐 모델 예측 함수가 클로저). LightGBM 은 **바이트 → sha256 → `Booster(model_str=...)`** 로만 연다 — 파일 경로 API 는 한글 절대경로를 열지 못한다. 예측 전 컬럼을 매니페스트 순서로 재정렬한다(LightGBM 은 순서가 틀려도 오류 없이 틀린 값을 낸다).
- **`serving/_core.py` 는 자동 생성 파일 — 직접 고치지 말 것.** `tools/build_serving_core.py` 가 s01·s02 의 전처리·피처 함수와 s10 의 `bundle_predict`·`manifest_core_sha256` 원문을 AST 로 추출한다.
  SPEC 에 있는 src 함수를 고치면 `run.sh build` 로 재생성해야 하고, `tests/test_serving_codegen.py` 가 동기화를 강제한다.
- 서빙 파이프라인(`serving/pipeline.py`)은 **이력에만** 정비(계측정지 보간)를 적용하고 계획 행(대상일)에는 전력을 두지 않는다 → 구조상 인과적. 배치 파이프라인은 전체 데이터 양방향 보간이라 08-29·07-13·07-15 가 미래 정보를 본다(테스트가 이 목록을 고정). 9월 테스트 구간은 서빙 리플레이가 제출 예측과 비트 단위로 같다.
- 게이트 휴무 오분류(8/2·8/3)는 고치지 않았다. 계획 휴무를 믿게 하는 수정(규칙·게이트 입력 제한·데이터 정제)은 모두 ERP 결측 가동일(7/13·7/15)에서 실패한다.
  서비스 제안은 '완전 휴무 확정일에만 거는 보정 옵션(기본 꺼짐)'이고 시제품은 `experiments/2026-09-25_gate_fix/serving_plan_override.py` 다(아직 `serving/` 에 넣지 않음).
- 입력 계약·오류 코드·배포 절차는 `serving/README.md`. 공휴일은 `serving/config/holidays_kr.json`(학습용 `HOLIDAYS_2021` 은 08-16 까지뿐), 커버리지 밖 대상일은 422.

**실험 기록 (`experiments/`)** — 제출물과 분리된 분석·도구 기록(피처 제거 확인, 게이트 수정안, 보고서 docx 편집, 포드 패키지). 색인은 `experiments/README.md`.
노트북 빌드·테스트·재현성 검사·마감 스캔 어디에도 들어가지 않는다. 실험 스크립트는 파이프라인을 import 하므로 **반드시 각 실험 폴더의 `_work/`(git 제외)로 chdir 한 뒤** import 한다
(저장소에서 import 하면 `outputs/` 의 표·그림 색인을 덮어쓴다). 보관 결과를 덮지 않고 재실행하려면 `KAMP_EXP_OUT`·`KAMP_EXP_FIG`.

## 주의

- 블라인드 평가: `finalize` 는 노트북 **셀 소스를 주석 포함 그대로** 스캔한다. 예시 경로를 주석에도 쓰지 말 것. `outputs/` 안에서 셸 작업 디렉터리를 옮겨 도구를 돌리면 도구 상태 파일이 생겨 스캔에 걸린다.
- 생성 파일·JSON 은 `write_bytes(...encode("utf-8"))` 로 쓴다(Windows `write_text` 는 CRLF·cp949 가 섞여 sha 가 달라진다). `.gitattributes` 가 `*.sh`·`*.py` 를 LF 로 고정한다(pod 실행용).
- s10 은 루트 `README.md`·`requirements.txt` 를 쓰지 않는다(사람이 관리하는 `00_README.md`·`requirements.txt` 보호). 6장 본문은 `outputs/report_ch6_draft.md` 로만 나간다.
- 제출 zip 에서 제외: `_dev_docs/`, `experiments/`, `outputs/models/`, 각 폴더 `CLAUDE.local.md`, 이 `CLAUDE.md`(개발 안내, 제외 권장), `.git/`, `.omc/`, `.cache/`, `__pycache__/`, `outputs/*.log`(설치 경로가 박히는데 스캔 대상이 아님), 빈 `outputs/baseline_repro/`. `video/`(약 950MB — 1부 6편·2부 22편·3부 서빙 7편)는 용량 때문에 별도 결정. `serving/` 은 zip 에 포함한다(없는 환경에서는 서빙 테스트 5개 모듈이 `importorskip` 으로 건너뛴다).
  git 추적 여부와 별개이므로 작업 폴더를 그대로 압축하지 말고 목록대로 뺀다(00_README '제출물과의 관계'와 같은 목록). 결과의 정직한 한계는 `03_RESULTS.md`, 증상별 대처는 `04_TROUBLESHOOTING.md`.
- 폴더별 `CLAUDE.local.md`(src·tests·tools·serving·outputs 등, gitignore 대상)에 그 폴더 전용 메모가 있다. 그 폴더를 고치기 전에 읽는다.
- `verify-head` 의 `ch6_model_bundles`(bundle_id·manifest sha)는 **같은 OS 에서만** 일치한다 — 매니페스트 `runtime` 에 OS 정보가 들어가 해시가 달라진다. OS 가 다르면 이 표는 반드시 다르고, 다른 표도 부동소수 차이로 달라질 수 있다(pod 실행은 미검증).
