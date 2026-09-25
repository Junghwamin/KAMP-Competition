# KAMP ⑤ 자원 최적화 — 작업 단축 명령
#
#   make setup    pod 셋업 (폰트 + 패키지 + 커널)
#   make check    환경 점검만 (실행 전 확인)
#   make run      노트북 배치 실행 (세션이 끊겨도 안전)
#   make test     검증 테스트 427개 (⚠️ outputs/ 를 FAST 로 덮어씀)
#   make build    src/ 에서 노트북 재조립 + serving/_core.py 재생성 (노트북 실행 출력이 지워짐)
#   make verify   재현성 검사 (동일 시드 2회 비교, ⚠️ outputs/ 를 FAST 로 덮어씀 · 같은 모드끼리 비교)
#   make verify-head   재현성 검사 (방금 실행 결과 vs 커밋된 결과)
#   make bundle-verify 모델 번들 무결성·자가검증
#   make replay   평가 번들로 9/1~9/14 재생 → .cache/replay_eval.csv
#   make serve    REST API 기동 (KAMP_BUNDLE_DIR · PORT, 기본 8000)
#   make serving-test  서빙 단위·코드생성 테스트 (outputs 안 건드림)
#   make serving-test-pipeline  번들·리플레이·API 테스트 (⚠️ outputs/ 를 FAST 로 덮어씀)
#   make clean    산출물 삭제 (data/ 와 소스는 보존)
#
# 모든 명령은 이 폴더를 CWD 로 실행해야 한다 (노트북이 상대경로를 쓴다).

NB       := 자원최적화_제안모델.ipynb
PY       := python -X utf8
SEEDENV  := PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8

.PHONY: setup check run run-fast test build verify verify-head finalize clean help bundle-verify replay serve serving-test serving-test-pipeline

help:
	@grep -E '^#   ' Makefile | sed 's/^#   //'

setup:
	bash setup_pod.sh

check:
	$(SEEDENV) $(PY) -c "import sys; sys.path.insert(0,'tools'); print('tools 로드 OK')"
	$(SEEDENV) $(PY) -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=120 START_HERE.ipynb

# 배치 실행. Jupyter UI 로 돌리다 세션이 끊기는 사고를 막는다.
# FULL 모드는 CPU 에서 30~60분 걸린다.
run:
	@mkdir -p outputs
	$(SEEDENV) $(PY) -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=7200 \
	  --ExecutePreprocessor.kernel_name=python3 $(NB) 2>&1 | tee outputs/run.log
	$(MAKE) finalize

# 빠른 확인용(수치는 최종값이 아니다). Optuna 3회, 에폭 5회로 축소.
run-fast:
	@mkdir -p outputs
	$(SEEDENV) KAMP_FAST=1 $(PY) -m nbconvert --to notebook --execute --inplace \
	  --ExecutePreprocessor.timeout=3600 $(NB) 2>&1 | tee outputs/run_fast.log
	$(MAKE) finalize

test:
	$(SEEDENV) KAMP_FAST=1 $(PY) -m pytest tests/ -q -m "not slow"

build:
	$(SEEDENV) $(PY) tools/build_notebook.py
	$(SEEDENV) $(PY) tools/build_serving_core.py

# 실행 후 필수. stderr 에 박힌 설치 경로(사용자명 포함)를 지우고 개인정보를 후스캔한다.
finalize:
	$(SEEDENV) $(PY) tools/finalize_notebook.py

verify:
	$(SEEDENV) KAMP_FAST=1 $(PY) tools/check_reproducibility.py

verify-head:
	$(SEEDENV) $(PY) tools/check_reproducibility.py --baseline-ref HEAD

# 개발용 기본값. 운영에서는 outputs 밖으로 복사한 불변 사본을 가리킨다(serving/README.md §3)
KAMP_BUNDLE_DIR ?= outputs/models/full/deploy
PORT            ?= 8000
bundle-verify:
	$(SEEDENV) $(PY) -m serving verify --bundle $(KAMP_BUNDLE_DIR)

# 평가 번들로 9월 테스트 구간 재생 — 제출 예측과 비트 단위로 같아야 한다(run.sh replay 와 동일)
replay:
	@mkdir -p .cache
	$(SEEDENV) $(PY) -m serving replay --bundle outputs/models/full/eval --data data/okm_augumented_2021.csv --start 2021-09-01 --end 2021-09-14 --out .cache/replay_eval.csv

serve:
	KAMP_BUNDLE_DIR=$(KAMP_BUNDLE_DIR) $(PY) -m uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port $(PORT)

serving-test:
	$(SEEDENV) KAMP_FAST=1 $(PY) -m pytest tests/test_serving_unit.py tests/test_serving_codegen.py -q

# ⚠️ FAST 파이프라인을 돌려 outputs/ 를 FAST 결과로 덮어쓴다(FULL 결과가 필요하면 run 을 다시)
serving-test-pipeline:
	$(SEEDENV) KAMP_FAST=1 $(PY) -m pytest tests/test_s10_bundles.py tests/test_serving_replay.py tests/test_serving_api.py -q -m "not slow"

clean:
	rm -rf outputs/figures outputs/tables outputs/models outputs/baseline_repro
	rm -f outputs/*.log outputs/*.csv outputs/*.pptx
	find outputs -maxdepth 1 -name '*.md' ! -name 'CLAUDE.local.md' -delete
	rm -rf .cache
	@echo "산출물 삭제 완료 (data/ · src/ · tests/ 는 보존)"
