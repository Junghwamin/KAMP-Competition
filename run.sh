#!/usr/bin/env bash
# make 가 없는 환경을 위한 대체 스크립트.
#
#   bash run.sh check      환경 점검
#   bash run.sh run        본실행 (FULL, 30~60분)
#   bash run.sh run-fast   빠른 확인 (3~5분, 수치는 최종값 아님)
#   bash run.sh finalize   실행 후 마감 (필수)
#   bash run.sh test       검증 테스트 427개 (⚠️ outputs/ 를 FAST 로 덮어씀)
#   bash run.sh build      src/ 에서 노트북 재조립 + serving/_core.py 재생성 (노트북 실행 출력이 지워짐)
#   bash run.sh verify     재현성 검사 (동일 시드 2회, ⚠️ outputs/ 를 FAST 로 덮어씀 · 같은 모드끼리 비교)
#   bash run.sh verify-head  재현성 검사 (방금 실행 결과 vs 커밋된 결과, 재실행 없음)
#   bash run.sh bundle-verify  모델 번들 무결성·자가검증 (outputs/models/full/deploy)
#   bash run.sh replay     평가 번들로 9/1~9/14 재생 → .cache/replay_eval.csv
#   bash run.sh serve      REST API 기동 (KAMP_BUNDLE_DIR, 기본 outputs/models/full/deploy · PORT, 기본 8000)
#   bash run.sh serving-test  서빙 단위·코드생성 테스트 (파이프라인 없음, outputs 안 건드림)
#   bash run.sh serving-test-pipeline  번들·리플레이·API 테스트 (⚠️ outputs/ 를 FAST 로 덮어씀)
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
  build)    "${PY[@]}" tools/build_notebook.py && "${PY[@]}" tools/build_serving_core.py ;;
  verify)   KAMP_FAST=1 "${PY[@]}" tools/check_reproducibility.py ;;
  verify-head) "${PY[@]}" tools/check_reproducibility.py --baseline-ref HEAD ;;
  bundle-verify) "${PY[@]}" -m serving verify --bundle "${KAMP_BUNDLE_DIR:-outputs/models/full/deploy}" ;;
  replay)
    mkdir -p .cache
    "${PY[@]}" -m serving replay --bundle outputs/models/full/eval       --data data/okm_augumented_2021.csv --start 2021-09-01 --end 2021-09-14 --out .cache/replay_eval.csv
    ;;
  serve)
    KAMP_BUNDLE_DIR="${KAMP_BUNDLE_DIR:-outputs/models/full/deploy}"       "${PY[@]}" -m uvicorn --factory serving.api:create_app --workers 1 --host 127.0.0.1 --port "${PORT:-8000}"
    ;;
  serving-test) KAMP_FAST=1 "${PY[@]}" -m pytest tests/test_serving_unit.py tests/test_serving_codegen.py -q ;;
  serving-test-pipeline)  # ⚠️ FAST 파이프라인을 돌려 outputs/ 를 FAST 결과로 덮어쓴다
    KAMP_FAST=1 "${PY[@]}" -m pytest tests/test_s10_bundles.py tests/test_serving_replay.py tests/test_serving_api.py -q -m "not slow" ;;
  *)        grep -E '^#   ' "$0" | sed 's/^#   //' ;;
esac
