"""4장 — 가이드북 베이스라인 재현 테스트.

핵심은 **버그가 실제로 재현되었는지**를 증명하는 것이다.
버그를 재현하지 못하면 보고서 2.2절의 "원본 vs 보정" 비교가 근거를 잃는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 4.1 Seasonal Naive
# ══════════════════════════════════════════════════════════════════════
def test_seasonal_naive_is_168h_lag(s04):
    """Seasonal Naive 예측값이 정확히 168시간 전 실측과 같다."""
    pred = s04.seasonal_naive_predict(s04.feat)
    t = pd.Timestamp("2021-09-05 14:00")
    assert float(pred.loc[t]) == pytest.approx(
        float(s04.feat.loc[t - pd.Timedelta(hours=168), "y_avg"])
    )


def test_naive_metrics_are_finite(s04):
    """Naive 지표가 모두 유한하고 합리적 범위다."""
    m = s04.naive_metrics
    assert all(np.isfinite(v) for v in m.values())
    assert 0 < m["MAE"] < 50
    assert m["RMSE"] >= m["MAE"]


# ══════════════════════════════════════════════════════════════════════
# 4.3 RF 원본 — iloc[0] 버그 증명 (보고서 2.2절의 핵심)
# ══════════════════════════════════════════════════════════════════════
def test_rf_original_has_exactly_one_effective_feature(s04):
    """`iloc[0]` 버그로 실질 피처가 **1개**뿐이다."""
    assert s04.rf_original["실질 피처 수"] == 1


def test_rf_importance_fully_concentrated(s04):
    """피처 중요도가 단 하나의 변수에 집중된다 (나머지는 상수라 기여 0)."""
    imp = s04.rf_importance
    assert float(imp.max()) > 0.99
    assert int((imp > 1e-6).sum()) == 1
    # 집중된 변수는 유일하게 행마다 변하는 '현재 15분값'이다
    assert s04.rf_feat_names[int(np.argmax(imp))] == "현재 15분값"


def test_rf_bug_signature_train_mse_exceeds_test(s04):
    """버그 서명: 학습 MSE > 테스트 MSE.

    과적합이 전혀 없다는 것은 모델이 학습할 것이 1개뿐이었다는 뜻이다.
    plan.md 가 보고한 185.98 / 183.06 과 같은 구조가 재현되어야 한다.
    """
    o = s04.rf_original
    assert o["학습 MSE"] > o["테스트 MSE"]
    assert o["과적합(학습<테스트)"] is False
    # 절대값도 plan.md 보고치 근방이어야 한다(정비로 소폭 달라질 수 있음)
    assert o["학습 MSE"] == pytest.approx(186.0, rel=0.05)
    assert o["테스트 MSE"] == pytest.approx(183.5, rel=0.05)


def test_corrected_rf_uses_all_features(s04):
    """보정 RF는 44개 피처를 모두 실제로 사용한다 (원본 1개와 대비)."""
    assert s04.rf_corrected["실질 피처 수"] == len(s04.FEATURE_COLS)
    assert s04.rf_corrected["실질 피처 수"] > 40


# ══════════════════════════════════════════════════════════════════════
# 4.4 무음 미반영 — 게이트 4
# ══════════════════════════════════════════════════════════════════════
def test_defect_table_has_13_entries(s04):
    """결함 13건이 빠짐없이 표로 정리된다 (보고서 2.2절)."""
    assert len(s04.defects_tbl) == 13
    cells = set(s04.defects_tbl["셀"])
    # plan.md 가 지목한 핵심 셀이 모두 포함되어야 한다
    for c in (4, 13, 21, 32, 34, 39, 48, 52, 73, 75, 80):
        assert c in cells, f"셀 {c} 결함이 누락됐다"


def test_chained_assignment_is_not_silent(s04):
    """가이드북의 연쇄 대입이 **조용히 넘어가지 않는다**.

    무음 미반영(합계 0)이거나 예외로 승격되어야 한다. 둘 중 어느 쪽이든
    "무오류 완주"만 보고 넘어가는 일이 없어야 한다.
    """
    row = s04.silent_noop_tbl[s04.silent_noop_tbl["항목"].str.contains("원본 패턴")].iloc[0]
    assert int(row["합계"]) == 0
    assert ("무음 미반영" in row["판정"]) or ("승격" in row["판정"])


def test_corrected_weekend_is_nonzero(s04):
    """보정 Weekend 합계 > 0 — 가이드북에서는 전부 0이 되는 값."""
    row = s04.silent_noop_tbl[s04.silent_noop_tbl["항목"].str.contains(r"Weekend \(보정")].iloc[0]
    assert int(row["합계"]) > 0
    # 토·일 시간 수와 일치해야 한다
    d = s04.df
    expected = int((d.index.dayofweek >= 5).sum())
    assert int(row["합계"]) == expected


def test_corrected_vacation_equals_holiday_hours(s04):
    """보정 Vacation 합계 == 공휴일 시간수."""
    row = s04.silent_noop_tbl[s04.silent_noop_tbl["항목"].str.contains("Vacation")].iloc[0]
    d = s04.df
    expected = int(d.index.normalize().isin(s04.HOLIDAYS_2021).sum())
    assert int(row["합계"]) == expected
    assert "일치" in row["판정"]


def test_gate4_passes(s04):
    """게이트 4 전 항목 통과."""
    assert s04.gate4["통과"].all()


# ══════════════════════════════════════════════════════════════════════
# 4.5 보정 조건
# ══════════════════════════════════════════════════════════════════════
def test_corrected_models_use_day_ahead_horizon(s04):
    """보정 모델은 전부 Day-ahead 지평이다 (원본 RNN은 1시간 앞)."""
    t = s04.corrected_tbl
    horizons = set(t.loc[t["예측지평"].notna(), "예측지평"])
    assert horizons == {"Day-ahead"}
    if "예측지평" in s04.rnn_original:
        assert s04.rnn_original["예측지평"] == "1시간 앞"


def test_corrected_rf_beats_naive(s04):
    """보정 RF가 Seasonal Naive 하한을 통과한다."""
    assert s04.rf_corrected["MAE"] < s04.naive_metrics["MAE"]


def test_corrected_predictions_have_336_rows(s04):
    """보정 모델 예측이 테스트 336시간을 모두 덮는다."""
    for name, pred in s04.baseline_preds.items():
        assert pred is not None, f"{name} 예측이 없다"
        assert len(pred) == 336, f"{name}: {len(pred)}행"
        assert np.isfinite(np.asarray(pred, float)).all(), f"{name}에 NaN/inf"


def test_corrected_metrics_reasonable(s04):
    """보정 모델 지표가 합리적 범위다 (FAST 모드 기준 느슨하게)."""
    t = s04.corrected_tbl
    for _, r in t.iterrows():
        if pd.isna(r.get("MAE")):
            continue
        assert 0 < r["MAE"] < 60, f"{r['모델']} MAE={r['MAE']}"
        assert r["RMSE"] >= r["MAE"]


def test_baseline_candidate_set_available(s04):
    """베이스라인 후보 집합이 6장 선정에 쓸 수 있게 준비되어 있다."""
    assert "Seasonal Naive" in s04.baseline_preds
    assert "Random Forest (보정)" in s04.baseline_preds
    assert len(s04.baseline_preds) >= 2


def test_figure_f16_exists(s04, project_root):
    """F16 중요도 집중 그림이 생성되었다 (버그 증명 그림)."""
    hits = list((project_root / "outputs" / "figures").glob("F16_*.png"))
    assert hits and hits[0].stat().st_size > 5000
