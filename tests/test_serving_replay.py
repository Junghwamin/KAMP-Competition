"""서빙 리플레이 테스트 — 번들 + 원시 입력만으로 배치 파이프라인 예측을 비트 단위로 재현한다.

서빙(`serving/`)은 src 모듈을 import 하지 않고, 이력(전날까지 실측) + 계획(대상일 ERP·기상)
만으로 피처를 다시 만든다. 이 파일은 그 경로가 노트북 배치 결과와 같은지 확인한다.

- 리플레이(원자료 첫날부터 이력)의 테스트 336시간 예측 == 5장 테스트 예측 (비트)
- 14일 창 리플레이 == 전체 이력 리플레이 (rolling std 두 개만 ~1e-13 오차 허용)
- 인과성 감사: 배치 피처가 **미래 정보를 쓰는 대상일**을 고정한다
  (2021-08-29 1일, 그리고 계획 시간값이 손상된 07-13·07-15)
- golden 자가검증 · CLI 종료코드 계약(0 정상 · 2 입력 오류 · 3 번들 오류)

모든 테스트는 `s10` fixture 에 의존한다(파이프라인 결과와 방금 쓴 FAST 번들).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# serving/ 이 없는 환경(노트북·데이터만 받은 경우)에서는 이 모듈을 통째로 건너뛴다
pytest.importorskip("serving")

SERVICE = "2단계 레짐(3분류)"
CLF = "피크 직접분류"
STD_COLS = ("roll24_avg_std", "roll168_avg_std")


def _compare_features(got: pd.DataFrame, want: pd.DataFrame, cols: list[str]) -> list[str]:
    """피처 비교 — std 두 개는 rtol 1e-9, 나머지는 정확 일치. 어긋난 컬럼 목록을 돌려준다."""
    assert len(got) == len(want) == 24
    bad = []
    for c in cols:
        g, w = got[c].to_numpy(dtype=float), want[c].to_numpy(dtype=float)
        ok = np.allclose(g, w, rtol=1e-9, atol=0) if c in STD_COLS else np.array_equal(g, w)
        if not ok:
            bad.append(c)
    return bad


def _ymd(ts) -> int:
    """Timestamp → YYYYMMDD 정수."""
    return int(pd.Timestamp(ts).strftime("%Y%m%d"))


@pytest.fixture(scope="module")
def bundle_dirs(s10, project_root) -> dict:
    """역할별 FAST 번들 디렉터리(s10 이 방금 썼다)."""
    base = project_root / "outputs" / "models" / "fast"
    return {"eval": base / "eval", "deploy": base / "deploy"}


@pytest.fixture(scope="module")
def bundle_eval(bundle_dirs):
    """서빙 로더로 연 평가 번들."""
    import serving

    return serving.load_bundle(bundle_dirs["eval"], allow_fast=True)


@pytest.fixture(scope="module")
def raw(s10) -> pd.DataFrame:
    """원자료(정비 전) 사본."""
    return s10.df_raw.copy()


@pytest.fixture(scope="module")
def anchored(bundle_eval, raw) -> pd.DataFrame:
    """원자료 첫날부터의 이력으로 09-01~09-14 를 하루씩 재생한 결과."""
    from serving.cli import replay

    return replay(bundle_eval, raw, "2021-09-01", "2021-09-14")


# ══════════════════════════════════════════════════════════════════════
# 1. 리플레이 == 배치 테스트 예측
# ══════════════════════════════════════════════════════════════════════
def test_replay_matches_batch_test_predictions(s10, anchored, project_root):
    """전체 이력 리플레이의 336시간 예측이 5장 테스트 예측과 비트 단위로 같다."""
    te_s, te_c = s10.test_results[SERVICE], s10.test_results[CLF]
    out = anchored
    assert len(out) == 336
    assert list(out["ts"]) == list(pd.DatetimeIndex(te_s["index"]).strftime("%Y-%m-%d %H:%M:%S"))
    assert np.array_equal(out["y_avg_pred"].to_numpy(), te_s["pred_avg"])
    assert np.array_equal(out["y_peak_pred"].to_numpy(), te_s["pred_peak"])
    assert np.array_equal(out["peak_prob"].to_numpy(), te_c["prob"])
    assert np.array_equal(out["q10"].to_numpy(), s10._unc["lo"])
    assert np.array_equal(out["q50"].to_numpy(), s10._unc["mid"])
    assert np.array_equal(out["q90"].to_numpy(), s10._unc["hi"])


def test_replay_labels_and_interval(s10, anchored, project_root):
    """경보 라벨 = y_peak_pred ≥ τ(제출 파일과 같음) · 예측구간 = 10.5절 반폭 규칙."""
    te_s = s10.test_results[SERVICE]
    tau = s10.cv_results[SERVICE]["tau"]
    assert np.array_equal(anchored["peak_label"].to_numpy(), (te_s["pred_peak"] >= tau).astype(int))
    if s10.FINAL_MODEL_NAME == s10.SERVICE_MODEL_NAME:
        sub = pd.read_csv(project_root / "outputs" / "predictions_test_336h.csv", encoding="utf-8-sig")
        assert np.array_equal(anchored["peak_label"].to_numpy(), sub["peak_pred_label"].to_numpy())
    X = s10.feat.loc[te_s["index"], s10.FEATURE_COLS]
    lo, hi = s10.interval_bounds(te_s["pred_avg"], X["is_shutdown"].to_numpy(), s10.BUNDLE_SUMMARY["halfwidths"])
    assert np.array_equal(anchored["pi_lo"].to_numpy(), lo)
    assert np.array_equal(anchored["pi_hi"].to_numpy(), hi)


# ══════════════════════════════════════════════════════════════════════
# 2. 14일 창 == 전체 이력
# ══════════════════════════════════════════════════════════════════════
def test_window14_replay_equals_anchored(bundle_eval, raw, anchored):
    """권장 이력 14일 창 리플레이의 예측이 전체 이력 리플레이와 같다."""
    from serving.cli import replay

    w14 = replay(bundle_eval, raw, "2021-09-01", "2021-09-14", window_days=14)
    assert list(w14.columns) == list(anchored.columns)
    assert list(w14["ts"]) == list(anchored["ts"])
    for c in anchored.columns.drop("ts"):
        assert np.array_equal(w14[c].to_numpy(), anchored[c].to_numpy()), c


def test_window14_features_match_batch(s10, bundle_eval, raw):
    """14일 이력으로 만든 09-14 피처 — std 두 개는 rtol 1e-9, 나머지 42개는 정확 일치."""
    from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS
    from serving.pipeline import build_day_ahead_features

    target = pd.Timestamp("2021-09-14")
    ymd = raw["날짜"].astype(int)
    hist = raw[(ymd >= _ymd(target - pd.Timedelta(days=14))) & (ymd <= _ymd(target - pd.Timedelta(days=1)))]
    plan = raw.loc[ymd == _ymd(target), PLAN_COLUMNS]
    cols = bundle_eval.columns
    X, info = build_day_ahead_features(
        hist[HISTORY_COLUMNS], plan, columns=cols, theta=bundle_eval.theta,
        holidays=bundle_eval.holidays_for(target), min_days=8,
    )
    assert info["history_days"] == 14
    want = s10.feat.loc[s10.feat.index.normalize() == target, cols]
    exact = [c for c in cols if c not in STD_COLS]
    assert len(exact) == 42
    assert _compare_features(X, want, cols) == []


# ══════════════════════════════════════════════════════════════════════
# 3. 인과성 감사 — 배치 피처가 미래 정보를 쓰는 날짜를 고정한다
# ══════════════════════════════════════════════════════════════════════
def test_causality_audit_jun_to_sep(s10, bundle_eval, raw):
    """06-01~09-14 각 대상일: (전날까지 이력, 대상일 계획) 피처 == 배치 피처.

    배치 파이프라인은 전체 기간을 한 번에 정비하므로, 이력 끝이 계측정지인 날은
    `limit_direction='both'` 보간이 대상일(미래) 실측으로 채운다. 그런 대상일이 정확히
    2021-08-29 하루임을 고정한다. 07-13·07-15 는 계획의 `시간` 값이 손상(70~188)돼
    서빙이 계획을 거부한다(PLAN_INVALID) — 배치는 그날 전체 24행을 보고 복원했다.
    """
    from serving.contract import PLAN_COLUMNS
    from serving.errors import ServingInputError
    from serving.pipeline import build_day_ahead_features

    ymd = raw["날짜"].astype(int)
    first = int(ymd.min())
    cols = bundle_eval.columns
    feat_day = s10.feat.index.normalize()
    mismatch: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for target in pd.date_range("2021-06-01", "2021-09-14", freq="D"):
        hist = raw[(ymd >= first) & (ymd <= _ymd(target - pd.Timedelta(days=1)))]
        plan = raw.loc[ymd == _ymd(target), PLAN_COLUMNS]
        try:
            X, _ = build_day_ahead_features(
                hist, plan, columns=cols, theta=bundle_eval.theta,
                holidays=bundle_eval.holidays_for(target), min_days=8,
            )
        except ServingInputError as e:
            errors[str(target.date())] = e.code
            continue
        bad = _compare_features(X, s10.feat.loc[feat_day == target, cols], cols)
        if bad:
            mismatch[str(target.date())] = bad
    assert set(mismatch) == {"2021-08-29"}, mismatch
    # 08-29 에 어긋나는 것은 직전일(계측정지로 끝난 이력)에 의존하는 피처뿐이다
    assert set(mismatch["2021-08-29"]) <= {
        "y_avg_lag24", "y_peak_lag24", "roll24_avg_mean", "roll24_avg_max", "roll24_avg_std",
        "roll24_peak_max", "roll24_peak_cnt", "roll168_avg_mean", "roll168_avg_max",
        "roll168_avg_std", "roll168_peak_max", "roll168_peak_cnt",
    }
    assert errors == {"2021-07-13": "PLAN_INVALID", "2021-07-15": "PLAN_INVALID"}


# ══════════════════════════════════════════════════════════════════════
# 4. golden 자가검증
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("role", ["eval", "deploy"])
def test_golden_selftest_passes(role, bundle_dirs):
    """두 번들 모두 golden 자가검증(피처 재생성 + 예측)을 통과한다."""
    import serving
    from serving.bundle import run_selftest

    b = serving.load_bundle(bundle_dirs[role], allow_fast=True)
    golden = json.loads((bundle_dirs[role] / "golden.json").read_bytes())
    res = run_selftest(b, golden)
    assert res == {"target_date": "2021-09-14", "features": "ok", "prediction": "ok"}
    assert len(golden["history"]["rows"]) == 21 * 24
    assert len(golden["plan"]["rows"]) == 24


# ══════════════════════════════════════════════════════════════════════
# 5. CLI — 종료코드 계약
# ══════════════════════════════════════════════════════════════════════
def _cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """`python -X utf8 -m serving ...` 를 프로젝트 루트에서 실행한다."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "serving", *args],
        cwd=str(cwd), env=env, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )


def test_cli_verify_exit_codes(bundle_dirs, project_root):
    """verify — FAST 허용 시 0, 허용하지 않으면 번들 오류 3(FAST_BUNDLE)."""
    ok = _cli(["verify", "--bundle", str(bundle_dirs["eval"]), "--allow-fast"], project_root)
    assert ok.returncode == 0, ok.stderr
    info = json.loads(ok.stdout)
    assert info["status"] == "ok" and info["role"] == "eval" and info["selftest"] == "ok"
    ng = _cli(["verify", "--bundle", str(bundle_dirs["eval"])], project_root)
    assert ng.returncode == 3
    assert "FAST_BUNDLE" in ng.stderr


def test_cli_replay_writes_336_rows(s10, bundle_dirs, project_root, tmp_path):
    """replay --out — 종료코드 0, 336행, 예측이 5장 테스트 예측과 같다."""
    out = tmp_path / "replay.csv"
    r = _cli(["replay", "--bundle", str(bundle_dirs["eval"]), "--allow-fast",
              "--data", "data/okm_augumented_2021.csv", "--start", "2021-09-01", "--end", "2021-09-14",
              "--out", str(out)], project_root)
    assert r.returncode == 0, r.stderr
    df = pd.read_csv(out, encoding="utf-8-sig", float_precision="round_trip")
    assert len(df) == 336
    assert np.allclose(df["y_avg_pred"], s10.test_results[SERVICE]["pred_avg"], rtol=0, atol=1e-9)


def _write_day_ahead_inputs(raw: pd.DataFrame, tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
    """이력(08-24~09-13, 21일)·계획(09-14) CSV 를 utf-8-sig 로 쓴다."""
    from serving.contract import HISTORY_COLUMNS, PLAN_COLUMNS

    ymd = raw["날짜"].astype(int)
    hist = raw.loc[(ymd >= 20210824) & (ymd <= 20210913), HISTORY_COLUMNS]
    plan = raw.loc[ymd == 20210914, PLAN_COLUMNS]
    hp, pp = tmp_path / "history.csv", tmp_path / "plan.csv"
    hist.to_csv(hp, index=False, encoding="utf-8-sig")
    plan.to_csv(pp, index=False, encoding="utf-8-sig")
    return hp, pp, plan


def test_cli_predict_matches_submission(s10, raw, bundle_dirs, project_root, tmp_path):
    """predict — 종료코드 0, 24행, 제출 파일 09-14 행과 4자리 반올림 일치."""
    hp, pp, _ = _write_day_ahead_inputs(raw, tmp_path)
    out = tmp_path / "pred.csv"
    r = _cli(["predict", "--bundle", str(bundle_dirs["eval"]), "--allow-fast",
              "--history", str(hp), "--plan", str(pp), "--out", str(out)], project_root)
    assert r.returncode == 0, r.stderr
    got = pd.read_csv(out, encoding="utf-8-sig", float_precision="round_trip")
    assert len(got) == 24
    assert got["ts"].str.startswith("2021-09-14").all()

    te_s = s10.test_results[SERVICE]
    day = pd.DatetimeIndex(te_s["index"]).normalize() == pd.Timestamp("2021-09-14")
    assert np.array_equal(np.round(got["y_avg_pred"].to_numpy(), 4), np.round(te_s["pred_avg"][day], 4))
    assert np.array_equal(np.round(got["y_peak_pred"].to_numpy(), 4), np.round(te_s["pred_peak"][day], 4))
    if s10.FINAL_MODEL_NAME == s10.SERVICE_MODEL_NAME:
        sub = pd.read_csv(project_root / "outputs" / "predictions_test_336h.csv",
                          encoding="utf-8-sig", float_precision="round_trip")
        sub = sub[sub["datetime"].str.startswith("2021-09-14")]
        assert len(sub) == 24
        assert list(got["ts"]) == list(sub["datetime"])
        assert np.array_equal(np.round(got["y_avg_pred"].to_numpy(), 4), sub["y_avg_pred"].to_numpy())
        assert np.array_equal(np.round(got["y_peak_pred"].to_numpy(), 4), sub["y_peak_pred"].to_numpy())
        assert np.array_equal(np.round(got["peak_prob"].to_numpy(), 6), sub["peak_prob"].to_numpy())
        assert np.array_equal(got["peak_label"].to_numpy(), sub["peak_pred_label"].to_numpy())


def test_cli_predict_rejects_power_in_plan(raw, bundle_dirs, project_root, tmp_path):
    """계획에 전력 컬럼('평균')이 있으면 입력 오류 — 종료코드 2(PLAN_HAS_POWER)."""
    hp, pp, plan = _write_day_ahead_inputs(raw, tmp_path)
    leaked = plan.assign(평균=100)
    leaked.to_csv(pp, index=False, encoding="utf-8-sig")
    r = _cli(["predict", "--bundle", str(bundle_dirs["eval"]), "--allow-fast",
              "--history", str(hp), "--plan", str(pp), "--out", str(tmp_path / "x.csv")], project_root)
    assert r.returncode == 2
    assert "PLAN_HAS_POWER" in r.stderr
    assert not (tmp_path / "x.csv").exists()
