"""2장 — 파생변수 구성 및 시간누수 차단 테스트.

핵심은 **누수 탐지기가 실제로 작동하는지**를 증명하는 것이다.
통과만 하는 검사는 아무것도 보장하지 않으므로, 일부러 누수를 주입해
탐지기가 반드시 실패하는지 확인하는 **음성 대조(negative control)** 를 둔다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 2.1 달력·공정상태
# ══════════════════════════════════════════════════════════════════════
def test_saturday_sunday_separated(s02):
    """토·일이 분리된 플래그다 (단일 '주말' 플래그 금지)."""
    assert "is_sat" in s02.FEATURE_COLS
    assert "is_sun" in s02.FEATURE_COLS
    f = s02.feat
    assert (f.loc[f.index.dayofweek == 5, "is_sat"] == 1).all()
    assert (f.loc[f.index.dayofweek == 6, "is_sun"] == 1).all()
    assert (f["is_sat"] & f["is_sun"]).sum() == 0


def test_daytime_matches_labor_cost_spec(s02):
    """주간 플래그 = 9≤h≤17 이며 `인건비==1.0` 구간과 정확히 일치한다."""
    f = s02.feat
    assert (f.loc[f["is_daytime"] == 1].index.hour >= 9).all()
    assert (f.loc[f["is_daytime"] == 1].index.hour <= 17).all()
    # 08시는 주간이 아니다 (인건비 1.5 구간)
    assert (f.loc[f.index.hour == 8, "is_daytime"] == 0).all()


def test_startup_flags(s02):
    """08시 기동·13시 재가동 플래그가 해당 시각에만 켜진다."""
    f = s02.feat
    assert set(f.loc[f["is_startup_08"] == 1].index.hour) == {8}
    assert set(f.loc[f["is_restart_13"] == 1].index.hour) == {13}


def test_shutdown_features_capture_summer_vacation(s02):
    """하계휴가 07-31~08-08이 휴무로, 08-09가 재가동일로 잡힌다.

    법정공휴일 기준이면 08-09 재가동을 잡을 수 없다.
    """
    f = s02.feat
    for d in ("2021-07-31", "2021-08-02", "2021-08-08"):
        assert (f.loc[d, "is_shutdown"] == 1).all(), f"{d}가 휴무로 안 잡힘"
    assert (f.loc["2021-08-09", "is_first_day_back"] == 1).all()
    # 휴무 n일차가 1..9로 증가
    nth = [int(f.loc[f"2021-0{m}-{dd:02d}", "shutdown_nth"].iloc[0])
           for m, dd in [(7, 31), (8, 1), (8, 2), (8, 8)]]
    assert nth == [1, 2, 3, 9]


def test_lag168_broken_by_summer_vacation(s02):
    """08-09 10시의 lag168 참조값(08-02 10시)이 휴무라 실측과 크게 다르다.

    보고서 1.6절 한계·7장 실패사례의 근거.
    """
    f = s02.feat
    actual = float(f.loc["2021-08-09 10:00", "y_avg"])
    lagged = float(f.loc["2021-08-09 10:00", "y_avg_lag168"])
    assert actual > 150 and lagged < 40, f"실측 {actual} vs lag168 {lagged}"


# ══════════════════════════════════════════════════════════════════════
# 2.2 원점 규칙
# ══════════════════════════════════════════════════════════════════════
def test_origin_stats_constant_within_day(s02):
    """원점 기준 이동통계는 **하루 안에서 값이 변하지 않는다**.

    같은 날 안에서 값이 변한다면 `rolling().shift(1)` 처럼 그날의 과거 시각을
    창에 넣고 있다는 뜻이고, 그것이 곧 누수다.
    """
    f = s02.feat
    roll_cols = [c for c in s02.FEATURE_COLS if c.startswith("roll")]
    assert roll_cols, "이동통계 피처가 없다"
    sub = f.loc[~f["is_warmup"], roll_cols]
    nuniq = sub.groupby(sub.index.normalize()).nunique()
    assert (nuniq <= 1).all().all(), "하루 안에서 이동통계가 변한다 → 누수"


def test_origin_stats_reference_previous_day(s02):
    """대상일 D+1 의 roll24 평균이 **D일 실제 평균**과 일치한다."""
    f = s02.feat
    target = pd.Timestamp("2021-06-15")
    prev = target - pd.Timedelta(days=1)
    got = float(f.loc[str(target.date()), "roll24_avg_mean"].iloc[0])
    expected = float(s02.df.loc[str(prev.date()), "y_avg"].mean())
    assert got == pytest.approx(expected, abs=1e-9)


def test_only_safe_lags_used(s02):
    """lag 는 24·48·168만 쓴다 (1~23시간 lag 금지)."""
    lag_cols = [c for c in s02.FEATURE_COLS if "_lag" in c]
    lags = {int(c.split("_lag")[1]) for c in lag_cols}
    assert lags == {24, 48, 168}


def test_lag24_matches_previous_day_same_hour(s02):
    """lag24 값이 전일 동시각 실측과 일치한다."""
    f = s02.feat
    t = pd.Timestamp("2021-06-15 14:00")
    assert float(f.loc[t, "y_avg_lag24"]) == pytest.approx(
        float(s02.df.loc[t - pd.Timedelta(hours=24), "y_avg"])
    )


# ══════════════════════════════════════════════════════════════════════
# 2.3 피처군
# ══════════════════════════════════════════════════════════════════════
def test_feature_groups_cover_report_table(s02):
    """보고서 2.3절 표의 6개 피처군이 모두 존재한다."""
    assert set(s02.FEATURE_GROUPS) == {
        "과거전력", "이동통계", "시간정보", "공정상태", "생산정보", "기상정보"
    }


def test_raw_duplicate_columns_excluded(s02):
    """`day`/`d`/`m` 은 피처에서 제외된다 (요일·일·월의 원본 중복)."""
    for c in ("day", "d", "m"):
        assert c not in s02.FEATURE_COLS


def test_target_columns_never_in_features(s02):
    """대상 시각의 전력값이 피처에 절대 들어가지 않는다."""
    for c in s02.ELECTRIC_COLS:
        assert c not in s02.FEATURE_COLS


def test_all_features_have_korean_labels(s02):
    """모든 피처에 보고서용 한글 라벨이 있다."""
    missing = [c for c in s02.FEATURE_COLS if c not in s02.FEATURE_LABELS]
    assert not missing, f"한글 라벨 누락: {missing}"


def test_cdd_hdd_definition(s02):
    """CDD = max(기온-24,0), HDD = max(18-기온,0)."""
    f = s02.feat
    assert np.allclose(f["cdd"], (f["기온"] - 24).clip(lower=0))
    assert np.allclose(f["hdd"], (18 - f["기온"]).clip(lower=0))


def test_warmup_rows_recorded(s02):
    """워밍업 탈락 행이 168행(168시간 창 미충족)으로 기록된다."""
    assert s02.N_WARMUP == 168
    f = s02.feat
    assert f.loc[~f["is_warmup"]].index.min() == pd.Timestamp("2021-01-08 00:00")


def test_no_missing_features_outside_warmup(s02):
    """워밍업 이후 구간에는 피처 결측이 없다."""
    f = s02.feat
    assert not f.loc[~f["is_warmup"], s02.FEATURE_COLS].isna().any().any()


def test_test_period_has_full_336_rows_with_features(s02):
    """테스트 구간 336시간이 전부 피처를 갖는다 (보간 선택의 근거).

    계측정지를 NaN 으로 뒀다면 rolling(168, min_periods=168) 이
    09-01~09-05의 120시간을 소실시켜 이 검사가 실패한다.
    """
    f = s02.feat
    test = f.loc[f.index >= s02.TEST_START]
    assert len(test) == 336
    assert not test[s02.FEATURE_COLS].isna().any().any()
    assert not test["is_warmup"].any()


# ══════════════════════════════════════════════════════════════════════
# 2.4 누수 검증 — 게이트 2
# ══════════════════════════════════════════════════════════════════════
def test_gate2_passes(s02):
    """게이트 2의 모든 점검이 통과했다."""
    assert s02.gate2["통과"].all()


def test_future_blind_equivalence_all_match(s02):
    """미래맹검 동등성 검사가 전 대상일에서 일치한다."""
    assert s02.blind_check["일치"].all()
    assert len(s02.blind_check) >= 12


# ── 음성 대조 — 탐지기가 실제로 누수를 잡는지 ────────────────────────
def _run_leak_probe(s02, inject):
    """누수 피처를 주입한 뒤 미래맹검 탐지기를 돌린다.

    `build_features` 를 임시 교체하되, 교체 대상 함수 안에서는 **원본**을 호출해야
    무한 재귀에 빠지지 않는다.
    """
    d, cal, theta = s02.df, s02.operating_calendar, s02.THETA
    orig_bf = s02.build_features  # 반드시 먼저 잡아둔다

    def leaky_build(dd, cc, th):
        out = orig_bf(dd, cc, th)
        out["LEAK"] = inject(dd, out)
        return out

    base = leaky_build(d, cal, theta)
    try:
        s02.FEATURE_COLS.append("LEAK")
        s02.build_features = leaky_build
        return s02.check_future_blind_equivalence(d, cal, theta, base, n_days=3)
    finally:
        s02.build_features = orig_bf
        s02.FEATURE_COLS.remove("LEAK")


def test_leak_detector_catches_same_hour_target(s02):
    """음성 대조 (1): 대상시각 전력을 피처로 넣으면 탐지기가 **반드시** 실패한다."""
    res = _run_leak_probe(s02, lambda dd, out: out["y_avg"])
    assert not res["일치"].all(), "누수를 주입했는데 탐지기가 통과시켰다"
    assert res.loc[~res["일치"], "불일치 피처"].str.contains("LEAK").all()


def test_leak_detector_catches_rolling_shift1(s02):
    """음성 대조 (2): 금지 패턴 `rolling(24).mean().shift(1)` 을 탐지한다.

    이 패턴은 h>0 에서 **같은 날의 과거 시각**을 창에 포함시켜 누수가 된다.
    참조 시각만 비교하는 타임스탬프 부기로는 잡히지 않으므로 미래맹검이 필요하다.
    """
    res = _run_leak_probe(
        s02, lambda dd, out: dd["y_avg"].rolling(24, min_periods=1).mean().shift(1)
    )
    assert not res["일치"].all(), "rolling().shift(1) 누수를 탐지하지 못했다"


def test_figure_f13_exists(s02, project_root):
    """F13 원점 경계 도식이 생성되었다."""
    hits = list((project_root / "outputs" / "figures").glob("F13_*.png"))
    assert hits and hits[0].stat().st_size > 5000
