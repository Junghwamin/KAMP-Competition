# KAMP ⑤ 자원 최적화 — 작업 단축 명령
#
#   make setup    pod 셋업 (폰트 + 패키지 + 커널)
#   make check    환경 점검만 (실행 전 확인)
#   make run      노트북 배치 실행 (세션이 끊겨도 안전)
#   make test     검증 테스트 236개
#   make build    src/ 에서 노트북 재조립
#   make verify   재현성 검사 (동일 시드 2회 비교)
#   make clean    산출물 삭제 (data/ 와 소스는 보존)
#
# 모든 명령은 이 폴더를 CWD 로 실행해야 한다 (노트북이 상대경로를 쓴다).

NB       := 자원최적화_제안모델.ipynb
PY       := python -X utf8
SEEDENV  := PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8

.PHONY: setup check run run-fast test build verify finalize clean help

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

# 실행 후 필수. stderr 에 박힌 설치 경로(사용자명 포함)를 지우고 개인정보를 후스캔한다.
finalize:
	$(SEEDENV) $(PY) tools/finalize_notebook.py

verify:
	$(SEEDENV) KAMP_FAST=1 $(PY) tools/check_reproducibility.py

clean:
	rm -rf outputs/figures outputs/tables outputs/models outputs/baseline_repro
	rm -f outputs/*.log outputs/*.csv outputs/*.md outputs/*.pptx
	rm -rf .cache
	@echo "산출물 삭제 완료 (data/ · src/ · tests/ 는 보존)"
