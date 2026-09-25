#!/usr/bin/env bash
# 노트북 배치 실행 — 세션이 끊겨도 안전하다.
#
#   bash run.sh                # FULL (CPU 30~60분)
#   KAMP_FAST=1 bash run.sh    # 빠른 동작 확인 (수치는 본실행 값이 아니다)
#
# 원본 노트북은 건드리지 않는다. 결과는 *_실행결과.ipynb 로 따로 저장된다.
set -euo pipefail
cd "$(dirname "$0")"

NB="자원최적화_실험전용.ipynb"
OUT="자원최적화_실험전용_실행결과.ipynb"
LOG="run.log"

[ -f "$NB" ] || { echo "노트북이 없다: $NB"; exit 1; }
[ -f "data/okm_augumented_2021.csv" ] || {
    echo "데이터가 없다: data/okm_augumented_2021.csv"
    echo "CSV 는 노트북 옆이 아니라 data/ 밑에 있어야 한다."; exit 1; }

mkdir -p outputs
echo "실행 시작 — 로그: $LOG   (FAST=${KAMP_FAST:-0})"

set +e
PYTHONHASHSEED=42 PYTHONIOENCODING=utf-8 \
python -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --output "$OUT" "$NB" > "$LOG" 2>&1
CODE=$?
set -e

echo "nbconvert 종료코드: $CODE" | tee -a "$LOG"
if [ "$CODE" -ne 0 ]; then
    echo
    echo "실패했다. 로그 끝부분:"
    tail -30 "$LOG"
    exit "$CODE"
fi

# 종료코드만 믿지 않는다 — 실제로 무엇이 생겼는지 센다.
python - <<'PY'
from pathlib import Path
import nbformat
nb = nbformat.read("자원최적화_실험전용_실행결과.ipynb", as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
ran = sum(1 for c in code if c.get("execution_count") is not None)
err = sum(1 for c in code
          if any(o.get("output_type") == "error" for o in c.get("outputs", [])))
tbl = len(list(Path("outputs/tables").glob("*.csv"))) if Path("outputs/tables").exists() else 0
fig = len(list(Path("outputs/figures").glob("*.png"))) if Path("outputs/figures").exists() else 0
pred = Path("outputs/predictions_test_336h.csv")
print(f"\n코드셀 {len(code)} · 실행됨 {ran} · 에러 {err}")
print(f"표 {tbl}개 · 그림 {fig}개 · 예측결과 {'있음' if pred.exists() else '없음'}")
if ran != len(code) or err or not pred.exists():
    raise SystemExit("완주하지 못했다 — run.log 를 확인할 것")
# 10.5절 모델 번들 — KAMP_SKIP_BUNDLE=1 로 건너뛰지 않았다면 두 개 다 있어야 한다
import os
mode = "fast" if os.environ.get("KAMP_FAST") == "1" else "full"
if os.environ.get("KAMP_SKIP_BUNDLE") != "1":
    missing = [r for r in ("eval", "deploy") if not Path(f"outputs/models/{mode}/{r}/manifest.json").exists()]
    if missing:
        raise SystemExit(f"모델 번들이 없다: outputs/models/{mode}/{missing}")
    print(f"모델 번들 {mode}/eval · {mode}/deploy 있음")
print("완주 확인")
PY

echo
echo "결과 노트북: 자원최적화_실험전용_실행결과.ipynb"
echo "산출물: outputs/figures · outputs/tables · outputs/predictions_test_336h.csv · outputs/models (모델 번들)"
