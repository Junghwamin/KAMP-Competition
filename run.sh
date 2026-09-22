#!/usr/bin/env bash
# make 가 없는 환경을 위한 대체 스크립트.
#
#   bash run.sh check      환경 점검
#   bash run.sh run        본실행 (FULL, 30~60분)
#   bash run.sh run-fast   빠른 확인 (3~5분, 수치는 최종값 아님)
#   bash run.sh finalize   실행 후 마감 (필수)
#   bash run.sh test       검증 테스트 236개
#   bash run.sh build      src/ 에서 노트북 재조립
#   bash run.sh verify     재현성 검사
set -euo pipefail
cd "$(dirname "$0")"

NB="자원최적화_제안모델.ipynb"
export PYTHONHASHSEED=42
export PYTHONIOENCODING=utf-8
PY=(python -X utf8)

case "${1:-help}" in
  check)
    "${PY[@]}" -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=300 START_HERE.ipynb
    echo "환경 점검 완료 — START_HERE.ipynb 의 마지막 셀 출력을 확인하라"
    ;;
  run)
    mkdir -p outputs
    "${PY[@]}" -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=7200 \
      --ExecutePreprocessor.kernel_name=python3 "$NB" 2>&1 | tee outputs/run.log
    bash "$0" finalize
    ;;
  run-fast)
    mkdir -p outputs
    KAMP_FAST=1 "${PY[@]}" -m nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=3600 "$NB" 2>&1 | tee outputs/run_fast.log
    bash "$0" finalize
    ;;
  finalize) "${PY[@]}" tools/finalize_notebook.py ;;
  test)     KAMP_FAST=1 "${PY[@]}" -m pytest tests/ -q -m "not slow" ;;
  build)    "${PY[@]}" tools/build_notebook.py ;;
  verify)   KAMP_FAST=1 "${PY[@]}" tools/check_reproducibility.py ;;
  *)        grep -E '^#   ' "$0" | sed 's/^#   //' ;;
esac
