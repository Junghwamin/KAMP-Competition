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
    python tools/check_reproducibility.py                    # 동일 시드 2회 실행 비교
    python tools/check_reproducibility.py --baseline-ref HEAD  # 방금 끝난 실행 vs 커밋된 결과

`--baseline-ref` 는 재실행하지 않는다. `git show <ref>:outputs/tables/*.csv` 를 기준으로
**지금 디스크의 결과**를 비교한다(실행시간 컬럼 제외). 코드를 고친 뒤 FULL 로 다시 돌렸을 때
"수치 변화 0" 을 증명하는 용도다. 제출 예측 파일은 바이트 단위로 비교한다.
기준과 현재가 같은 모드(FULL/FAST)여야 의미가 있다.
"""
from __future__ import annotations

import argparse
import io
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
    # 10.5절 모델 번들의 파일별 sha256 — 2회 실행 시 **모델 바이트까지** 같은지 증명한다
    "ch6_model_bundles",
    # 6.6절 지연변수 제거 효과 분해 (보고서 표 2-6 과 3.5절 근거)
    "ch2_lag_variants",
    "ch2_lag_decomposition",
    "ch2_lofo",
    "ch2_lag_d2",
    "ch2_gate_diagnosis",
]
PREDICTION_FILE = "outputs/predictions_test_336h.csv"
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


def _snapshot_reader(name: str) -> pd.DataFrame | None:
    """1회차 스냅샷에서 표를 읽는다(없으면 None)."""
    path = SNAPSHOT_DIR / f"{name}.csv"
    return pd.read_csv(path, encoding="utf-8-sig") if path.exists() else None


# 기준 커밋에 없어도 되는 표 — 이번에 새로 생긴 표만 명시한다(오타 ref 가 전부 "신규"로 통과하지 않게)
NEW_TABLES = {
    "ch6_model_bundles",
    "ch2_lag_variants", "ch2_lag_decomposition", "ch2_lofo", "ch2_lag_d2", "ch2_gate_diagnosis",
}


def _git_ok(*args: str) -> bool:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True).returncode == 0


def _git_show(ref: str, relpath: str) -> bytes | None:
    """`git show <ref>:<relpath>` 의 바이트. 그 커밋에 파일이 없으면 None.

    git 자체가 실패하면(ref 오류 등) 예외 — '파일 없음' 과 구분한다.
    """
    if not _git_ok("cat-file", "-e", f"{ref}:{relpath}"):
        return None
    proc = subprocess.run(
        ["git", "show", f"{ref}:{relpath}"], cwd=PROJECT_ROOT, capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git show 실패: {ref}:{relpath}")
    return proc.stdout


def _git_reader(ref: str):
    """커밋된 결과에서 표를 읽는 함수를 만든다."""
    def read(name: str) -> pd.DataFrame | None:
        raw = _git_show(ref, f"outputs/tables/{name}.csv")
        return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig") if raw is not None else None
    return read


def compare(read_baseline=_snapshot_reader, new_ok_if_absent: bool = False) -> pd.DataFrame:
    """기준(1회차 스냅샷 또는 커밋) 결과와 현재 결과를 비교한다."""
    rows = []
    for name in TARGET_TABLES:
        a = read_baseline(name)
        b_path = TBL_DIR / f"{name}.csv"
        if a is None and new_ok_if_absent and name in NEW_TABLES and b_path.exists():
            rows.append({"표": name, "일치": True, "사유": "기준에 없음(신규 표)", "비교 셀 수": 0})
            continue
        if a is None or not b_path.exists():
            rows.append({"표": name, "일치": False, "사유": "파일 없음", "비교 셀 수": 0})
            continue
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


def compare_with_ref(ref: str) -> int:
    """커밋된 결과(ref)와 현재 디스크 결과를 비교한다 — 재실행하지 않는다."""
    if not _git_ok("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"):
        print(f"[FAIL] git ref 를 찾을 수 없다: {ref}", file=sys.stderr)
        return 2
    result = compare(_git_reader(ref), new_ok_if_absent=True)
    base_pred = _git_show(ref, PREDICTION_FILE)
    cur_pred = (PROJECT_ROOT / PREDICTION_FILE).read_bytes()
    # 줄바꿈은 OS 마다 다르게 쓰인다(pandas to_csv = os.linesep) → 정규화 후 바이트 비교
    pred_ok = base_pred is not None and base_pred.replace(b"\r\n", b"\n") == cur_pred.replace(b"\r\n", b"\n")
    result = pd.concat([result, pd.DataFrame([{
        "표": "predictions_test_336h.csv (바이트)", "일치": pred_ok,
        "사유": "" if pred_ok else "바이트 불일치", "비교 셀 수": 0,
    }])], ignore_index=True)
    out = PROJECT_ROOT / ".cache" / f"reproducibility_vs_{ref.replace('/', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"── 커밋 {ref} 대비 현재 결과 (소수 {DECIMALS}자리, 실행시간 제외) ──")
    print(result.to_string(index=False))
    failed = result[~result["일치"]]
    if len(failed):
        print(f"\n[FAIL] {len(failed)}개 항목이 기준과 다르다", file=sys.stderr)
        return 1
    print(f"\n  ✅ 기준 {ref} 과 수치 변화 0 · 예측 파일 바이트 동일")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-ref", help="재실행 없이 이 git ref 의 결과와 비교한다 (예: HEAD)")
    args = ap.parse_args(argv)
    if args.baseline_ref:
        return compare_with_ref(args.baseline_ref)

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
