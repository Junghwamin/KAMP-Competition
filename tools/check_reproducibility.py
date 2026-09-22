"""동일 시드 2회 실행 결과가 일치하는지 검증한다 (보고서 6장 재현성 근거).

plan.md §6 검증항목 8: "동일 시드 2회 실행 결과가 소수 6자리까지 일치".

방법
----
1. 현재 `outputs/tables/` 의 주요 지표표를 스냅샷으로 보관한다.
2. 전 파이프라인을 다시 실행한다(동일 시드).
3. 두 결과를 수치 컬럼 기준으로 비교해 `reproducibility_check.csv` 를 만든다.

비교 대상에서 **실행시간(timing)** 은 제외한다. 벽시계 시간은 본질적으로
재현되지 않으며, 재현성 요건의 대상도 아니다.

사용법
------
    python build/check_reproducibility.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TBL_DIR = PROJECT_ROOT / "outputs" / "tables"
SNAPSHOT_DIR = PROJECT_ROOT / ".cache" / "repro_run1"
DECIMALS = 6

# 재현되어야 하는 핵심 지표표. 실행시간·환경 정보는 제외한다.
TARGET_TABLES = [
    "ch2_regression",
    "ch2_peak_detection",
    "ch2_peak_detection_oof",
    "ch2_ablation",
    "ch2_scorecard",
    "ch2_significance",
    "ch2_d1_vs_d2",
    "ch3_importance",
    "ch3_condition_mae",
    "ch3_rules",
    "ch4_levers",
    "ch4_scenarios",
    "ch5_calibration",
    "ch5_uncertainty",
    "ch1_theta",
    "ch3_folds",
]
EXCLUDE_TABLES = {"timing", "env_versions"}
# 벽시계 시간은 본질적으로 재현되지 않으며 재현성 요건의 대상도 아니다.
# 표 단위가 아니라 **컬럼 단위로** 제외해야 한다 — 학습시간이 성능표 안에 섞여 있다.
EXCLUDE_COL_TOKENS = ("시간(초)", "초)", "_sec", "elapsed")


def _is_timing_col(name: str) -> bool:
    """실행시간 컬럼인지 판정한다."""
    return any(tok in str(name) for tok in EXCLUDE_COL_TOKENS)


def snapshot() -> int:
    """현재 결과표를 1회차 스냅샷으로 보관한다."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for name in TARGET_TABLES:
        src = TBL_DIR / f"{name}.csv"
        if src.exists():
            shutil.copy2(src, SNAPSHOT_DIR / src.name)
            n += 1
    return n


def rerun() -> int:
    """동일 시드로 파이프라인을 재실행한다."""
    env_cmd = (
        "import sys; sys.path.insert(0, 'src'); import s10_package"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", env_cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def compare() -> pd.DataFrame:
    """1회차 스냅샷과 2회차 결과를 비교한다."""
    rows = []
    for name in TARGET_TABLES:
        a_path, b_path = SNAPSHOT_DIR / f"{name}.csv", TBL_DIR / f"{name}.csv"
        if not a_path.exists() or not b_path.exists():
            rows.append({"표": name, "일치": False, "사유": "파일 없음", "비교 셀 수": 0})
            continue
        a = pd.read_csv(a_path, encoding="utf-8-sig")
        b = pd.read_csv(b_path, encoding="utf-8-sig")
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            rows.append({"표": name, "일치": False, "사유": "형태/컬럼 불일치", "비교 셀 수": 0})
            continue

        compare_cols = [c for c in a.columns if not _is_timing_col(c)]
        skipped = [c for c in a.columns if _is_timing_col(c)]
        num_cols = [
            c for c in compare_cols if pd.api.types.is_numeric_dtype(a[c])
        ]
        obj_cols = [c for c in compare_cols if c not in num_cols]

        num_ok = True
        n_cells = 0
        max_diff = 0.0
        for c in num_cols:
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(np.nan_to_num(x) - np.nan_to_num(y)))
            max_diff = max(max_diff, float(d.max()) if len(d) else 0.0)
            num_ok &= bool((np.round(d, DECIMALS) == 0).all())
            n_cells += len(x)

        obj_ok = all(a[c].astype(str).equals(b[c].astype(str)) for c in obj_cols)
        n_cells += sum(len(a[c]) for c in obj_cols)

        rows.append(
            {
                "표": name,
                "일치": bool(num_ok and obj_ok),
                "사유": "" if (num_ok and obj_ok) else ("수치 불일치" if not num_ok else "문자 불일치"),
                "비교 셀 수": n_cells,
                "최대 절대차": round(max_diff, 10),
                "제외 컬럼(실행시간)": ", ".join(skipped) if skipped else "-",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    print(f"── 1회차 결과 스냅샷 ({snapshot()}개 표) ──")
    print("── 동일 시드로 2회차 실행 중 ──")
    if rerun() != 0:
        print("[FAIL] 2회차 실행이 실패했다", file=sys.stderr)
        return 1

    result = compare()
    result.to_csv(TBL_DIR / "reproducibility_check.csv", index=False, encoding="utf-8-sig")
    print(f"\n── 재현성 비교 (소수 {DECIMALS}자리) ──")
    print(result.to_string(index=False))

    failed = result[~result["일치"]]
    if len(failed):
        print(f"\n[FAIL] {len(failed)}개 표가 재현되지 않았다", file=sys.stderr)
        return 1
    total = int(result["비교 셀 수"].sum())
    print(f"\n  ✅ {len(result)}개 표 · {total:,}개 셀이 소수 {DECIMALS}자리까지 일치")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
