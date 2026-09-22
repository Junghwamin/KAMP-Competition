"""1장 — 데이터 이해 및 진단 오라클 테스트.

기대값은 `plan.md` §1 검증표와 프로젝트 메모리의 실측 결과에서 온다.
여기서 재유도하지 않는다.

주의: 원자료 기준 오라클과 **정비 후** 기준 오라클을 구분한다.
계측정지 17행을 보간하므로 요일별 평균 등 일부 통계는 소폭 달라진다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

POWER_COLS = ["15분", "30분", "45분", "60분"]


# ══════════════════════════════════════════════════════════════════════
# 1.1 적재 — 원자료 오라클
# ══════════════════════════════════════════════════════════════════════
def test_raw_shape(s01):
    """원자료는 6,168행 × 18열이다."""
    assert s01.df_raw.shape == (6168, 18)


def test_raw_no_bom_in_columns(s01):
    """BOM이 제거되어 첫 컬럼명이 '날짜'다 (utf-8-sig 사용 증명)."""
    assert s01.df_raw.columns[0] == "날짜"
    assert not any(c.startswith("﻿") for c in s01.df_raw.columns)


def test_raw_period(s01):
    """2021-01-01 ~ 09-14, 257일."""
    assert s01.df_raw["날짜"].min() == 20210101
    assert s01.df_raw["날짜"].max() == 20210914
    assert s01.df_raw["날짜"].nunique() == 257


def test_raw_missing_counts(s01):
    """결측: 풍속 3건, 강수량 1건, 공장인원 17건."""
    na = s01.df_raw.isna().sum()
    assert na["풍속"] == 3
    assert na["강수량"] == 1
    assert na["공장인원"] == 17
    assert na.drop(["풍속", "강수량", "공장인원"]).sum() == 0


def test_raw_key_duplicates(s01):
    """(날짜,시간) 중복 5건, 행 단위 완전중복은 0건."""
    assert int(s01.df_raw.duplicated(["날짜", "시간"]).sum()) == 5
    assert int(s01.df_raw.duplicated().sum()) == 0, "행 단위 중복이 0이라 duplicated()로는 복제를 못 잡는다"


def test_raw_damaged_hours(s01):
    """시간값 70~188인 48행이 7/13·7/15에 각 24행씩 있다."""
    raw = s01.df_raw
    bad = raw[(raw["시간"] < 0) | (raw["시간"] > 23)]
    assert len(bad) == 48
    assert bad["시간"].min() == 70 and bad["시간"].max() == 188
    assert bad.groupby("날짜").size().to_dict() == {20210713: 24, 20210715: 24}


def test_raw_variable_dict_covers_all_columns(s01):
    """변수사전이 18개 컬럼을 모두 설명한다."""
    assert len(s01.var_dict) == 18
    assert set(s01.var_dict["변수"]) == set(s01.df_raw.columns)
    assert (s01.var_dict["구분"] != "-").all(), "의미가 미기재된 컬럼이 있다"


# ══════════════════════════════════════════════════════════════════════
# 1.2 품질 진단 — 프로파일 복제 / 행순서 검증
# ══════════════════════════════════════════════════════════════════════
def test_profile_duplicates(s01):
    """고유 프로파일 142/257, 중복그룹 45개, 잉여 115일(44.7%)."""
    assert s01.N_UNIQUE_PROFILES == 142
    assert s01.N_TOTAL_DAYS == 257
    assert s01.N_DUP_GROUPS == 45
    assert s01.N_SURPLUS_DAYS == 115
    assert s01.N_SURPLUS_DAYS / s01.N_TOTAL_DAYS == pytest.approx(0.447, abs=0.001)


def test_profile_group_size_distribution(s01):
    """그룹 크기 분포 {2:12, 3:16, 4:11, 5:2, 6:2, 7:1, 15:1}."""
    dist = s01.profile_dup["일수"].value_counts().sort_index().to_dict()
    assert dist == {2: 12, 3: 16, 4: 11, 5: 2, 6: 2, 7: 1, 15: 1}


def test_largest_duplicate_group_has_wildly_different_temperature(s01):
    """최대 그룹 15일은 전력이 동일한데 기온은 -3.4 ~ 19.2°C로 전혀 다르다.

    실재하는 근무 패턴이 아니라 증강 산물이라는 핵심 증거.
    """
    top = s01.profile_dup.iloc[0]
    assert top["일수"] == 15
    assert top["기온_최저"] == pytest.approx(-3.4, abs=0.05)
    assert top["기온_최고"] == pytest.approx(19.2, abs=0.05)


def test_row_order_verified_by_temperature(s01):
    """손상일의 자정 경계 기온 점프가 전체 90분위 이내이고 일주기가 정상이다."""
    chk = s01.rowlayout_check
    assert len(chk) == 2
    p90 = chk.attrs["전체경계_90분위"]
    assert chk["직전일_경계점프"].max() <= p90
    assert chk["직후일_경계점프"].max() <= p90
    # 기온 최저는 새벽(행 2~7), 최고는 오후(행 11~16)
    assert chk["기온최저_행위치"].between(2, 7).all()
    assert chk["기온최고_행위치"].between(11, 16).all()


# ══════════════════════════════════════════════════════════════════════
# 1.3 정비 — 게이트 1
# ══════════════════════════════════════════════════════════════════════
def test_gate1_all_pass(s01):
    """게이트 1의 모든 점검이 통과했다."""
    assert s01.gate1["통과"].all()


def test_repair_resolves_duplicates_and_hours(s01):
    """정비 후 (날짜,시간) 중복 0, 시간 0~23 100%, 결측 0."""
    d = s01.df
    assert int(d.reset_index().duplicated(["날짜", "시간"]).sum()) == 0
    assert d["시간"].between(0, 23).all()
    assert int(d.isna().sum().sum()) == 0


def test_integrity_relation_is_half_up_rounding(s01):
    """무결성 산식은 `평균 = floor(4구간평균 + 0.5)` 이며 100% 일치한다.

    numpy/pandas 의 `round()` 는 은행가 반올림(ties-to-even)이라 87.6%만 맞는다.
    실제 산식은 half-up 이다.
    """
    d = s01.df
    m4 = d[POWER_COLS].mean(axis=1)
    assert (d["평균"] == np.floor(m4 + 0.5)).all()


def test_outage_block(s01):
    """센서측 계측정지 = 08-28 18시 ~ 08-29 10시 연속 17시간."""
    d = s01.df
    idx = d.index[d["is_outage"]]
    assert len(idx) == 17
    assert idx.min() == pd.Timestamp("2021-08-28 18:00")
    assert idx.max() == pd.Timestamp("2021-08-29 10:00")
    # 연속 블록인지 확인
    assert (idx.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_erp_missing_block(s01):
    """ERP측 결측 = 7/13·7/15 48시간, 생산량·공장인원 0, 인건비 전부 1.5."""
    d = s01.df
    m = d["is_erp_missing"]
    assert int(m.sum()) == 48
    dates = sorted(set(d.index[m].normalize()))
    assert dates == [pd.Timestamp("2021-07-13"), pd.Timestamp("2021-07-15")]
    raw = s01.df_raw
    sub = raw[raw["날짜"].isin([20210713, 20210715])]
    assert sub["생산량"].sum() == 0
    assert (sub["인건비"] == 1.5).all()
    # 전력은 정상 가동 수준 (센서측 정지와 대칭)
    assert d.loc[m, "y_avg"].mean() > 100


def test_outage_power_is_interpolated_not_nan(s01):
    """계측정지 구간은 NaN이 아니라 보간값이다.

    NaN 이면 rolling(168, min_periods=168) 이 테스트 구간 336h 중 120h(35.7%)를
    소실시켜 "336행·결측 0" 요건과 충돌한다.
    """
    d = s01.df
    assert d.loc[d["is_outage"], "y_avg"].notna().all()
    assert (d.loc[d["is_outage"], "y_avg"] > 0).all()


# ══════════════════════════════════════════════════════════════════════
# 1.4 가동 캘린더
# ══════════════════════════════════════════════════════════════════════
def test_calendar_detects_summer_shutdown(s01):
    """하계휴가 07-31 ~ 08-08 9일 연속 휴무를 포착한다 (가이드북 리스트에 없음)."""
    ls = s01.long_shutdowns
    hit = ls[(ls["시작"].astype(str) == "2021-07-31") & (ls["종료"].astype(str) == "2021-08-08")]
    assert len(hit) == 1
    assert int(hit.iloc[0]["일수"]) == 9
    assert not bool(hit.iloc[0]["법정공휴일포함"])


def test_baseline_holiday_list_missing_summer_vacation(s01):
    """가이드북 공휴일 리스트의 확실한 결함은 **하계휴가 9일 누락**이다."""
    assert not any(
        str(d).startswith("2021-07-3") or str(d).startswith("2021-08-0")
        for d in s01.BASELINE_HOLIDAYS_2021
    ), "하계휴가가 베이스라인 리스트에 이미 있다면 이 진단은 무의미하다"


def test_0211_and_0301_are_真_holidays_with_overwritten_power(s01):
    """02-11·03-01은 전력만 보면 가동일 같지만 실제로는 휴일이다.

    plan.md 발견 5는 이 두 날을 "정상 가동인데 휴일 표기"로 판정했으나,
    생산량·공장인원이 0이고 프로파일 복제그룹에 속하므로 실제로는 공휴일이
    맞고 **전력 채널이 증강으로 덮어씌워진** 것이다. 전력 단일 채널 판정의 위험.
    """
    ha = s01.holiday_audit.set_index("날짜")
    for d in ("2021-02-11", "2021-03-01"):
        assert ha.loc[d, "일생산량"] == 0, "생산량이 0이면 가동일이 아니다"
        assert bool(ha.loc[d, "증강불일치"]), "증강 채널 불일치로 분류되어야 한다"
        assert ha.loc[d, "판정"] == "휴일 맞음 (전력만 증강 덮어쓰기)"
    assert ha.loc["2021-02-11", "일평균전력"] == pytest.approx(127.3, abs=0.5)
    assert ha.loc["2021-03-01", "일평균전력"] == pytest.approx(129.0, abs=0.5)


def test_augmentation_channel_inconsistency(s01):
    """증강 채널 불일치 9일 — 생산량 0인데 전력이 평일 수준.

    ERP 결측(7/13·7/15)과 달리 `인건비`가 정상이고 대부분 일요일·법정공휴일이다.
    발견 1(프로파일 복제)의 독립적인 두 번째 증거.
    """
    d = s01.df
    days = sorted({str(t.date()) for t in d.index[d["is_aug_inconsistent"]]})
    assert days == [
        "2021-01-10", "2021-01-24", "2021-02-11", "2021-03-01", "2021-03-07",
        "2021-03-14", "2021-03-21", "2021-03-28", "2021-05-09",
    ]
    # 7/13·7/15 와 상호배타
    assert not (d["is_aug_inconsistent"] & d["is_erp_missing"]).any()
    # 이 날들은 생산량·공장인원이 0인데 전력은 가동 수준
    sub = d[d["is_aug_inconsistent"]]
    assert sub["생산량"].sum() == 0
    assert sub["y_avg"].mean() > 70


def test_sunday_power_anomaly_confirms_augmentation(s01):
    """일요일 37일 중 8일이 일평균전력 ≥70kW — 중앙값 22.9kW 대비 이상치."""
    cal = s01.operating_calendar
    sun = cal[cal["요일"] == 6]
    assert len(sun) == 37
    assert int((sun["일평균전력"] >= 70).sum()) == 8
    assert float(sun["일평균전력"].median()) == pytest.approx(22.9, abs=1.0)


def test_erp_missing_days_not_marked_shutdown(s01):
    """ERP 결측일은 생산량이 0이지만 휴무로 분류하면 안 된다 (전력 136 = 가동 중)."""
    cal = s01.operating_calendar
    for d in ("2021-07-13", "2021-07-15"):
        assert not bool(cal.loc[pd.Timestamp(d), "is_shutdown"])


# ══════════════════════════════════════════════════════════════════════
# 1.5 EDA
# ══════════════════════════════════════════════════════════════════════
def test_eda_headline_ratios(s01):
    """생산량 0 = 43.1%, 평균전력<30 = 33.2%."""
    d = s01.df
    assert float((d["생산량"] == 0).mean()) == pytest.approx(0.431, abs=0.003)
    assert float((d["y_avg"] < 30).mean()) == pytest.approx(0.332, abs=0.003)


def test_eda_correlations(s01):
    """corr(생산량,전력)=0.52, corr(기온,전력)=0.05."""
    d = s01.df
    assert float(d["생산량"].corr(d["y_avg"])) == pytest.approx(0.518, abs=0.01)
    assert float(d["기온"].corr(d["y_avg"])) == pytest.approx(0.053, abs=0.01)


def test_eda_weekday_vs_weekend(s01):
    """평일 103~123, 토 ~51, 일 ~41."""
    dow = s01.eda_dow["평균전력"]
    assert dow.loc[["월", "화", "수", "목", "금"]].between(103, 124).all()
    assert dow.loc["토"] == pytest.approx(51, abs=1.5)
    assert dow.loc["일"] == pytest.approx(41, abs=1.5)


def test_eda_hour_transitions(s01):
    """8시 기동·13시 재가동 증가, 12시·17시 감소."""
    h = s01.eda_hourly["평일"]
    assert h[8] > h[7] * 1.25, "8시 동시 기동 급등"
    assert h[12] < h[11] * 0.75, "12시 점심 급감"
    assert h[13] > h[12] * 1.4, "13시 재가동 급등"
    assert h[17] < h[16], "17시 이후 감소"


def test_peak_rate_hour8_is_highest(s01):
    """08시 피크율이 1위다 — L1 레버(기동 분산)의 근거."""
    pr = s01.eda_peak_hour["피크율"]
    assert pr.idxmax() == 8
    assert pr[8] > 24.0
    assert pr[12] == 0.0, "점심시간 피크 0건"


def test_monthly_max_peak(s01):
    """월별 최대 peak15 = 222/198/222/199/199/222/222/218/204.

    9월(테스트)이 204로 7월 222보다 낮다 → 9월만으로는 기본요금 절감 0원.
    """
    mm = s01.eda_monthly["최대peak15"].tolist()
    assert mm == [222, 198, 222, 199, 199, 222, 222, 218, 204]


def test_night_peaks_all_from_erp_missing(s01):
    """야간(19시+) 피크 7건이 전부 ERP 결측일에 귀속된다.

    → 보고서 1.3절 "야간 피크 0건"은 "정상 가동일 기준"으로 수정해야 한다.
    이 문장이 3.6절 규칙과 4장 L4 레버의 근거다.
    """
    night = s01.eda_night
    assert len(night) == 7
    assert bool(night["is_erp_missing"].all())


# ══════════════════════════════════════════════════════════════════════
# 1.6 피크 정의·임계값
# ══════════════════════════════════════════════════════════════════════
def test_theta_is_187(s01):
    """θ = 187 고정."""
    assert s01.THETA == 187.0


def test_peak15_definition_reproduces_report(s01):
    """`peak15 = max(4구간)` 에서만 187/201/222 가 재현된다."""
    tc = s01.theta_compare.set_index("정의")
    row = tc.loc["peak15 = max(15,30,45,60분)"]
    assert (row["상위5%(q95)"], row["상위1%(q99)"], row["최대값"]) == (187.0, 201.0, 222)


def test_single_15min_column_does_not_match(s01):
    """`15분` 단독은 173.x/188/207 로 보고서와 불일치한다 — 따라가면 안 된다."""
    tc = s01.theta_compare.set_index("정의")
    row = tc.loc["15분 단독"]
    assert row["상위5%(q95)"] == pytest.approx(173.5, abs=0.3)
    assert row["상위1%(q99)"] == 188.0
    assert row["최대값"] == 207
    assert row["최대값"] != 222, "15분 단독 정의는 보고서 최대값과 다르다"


def test_max_peak_occurrences(s01):
    """최대 222 도달 4건 = 01-29 08시, 03-29 08시, 06-29 08시, 07-19 11시."""
    d = s01.df
    hits = d.index[d["y_peak"] == 222]
    assert [str(t) for t in hits] == [
        "2021-01-29 08:00:00",
        "2021-03-29 08:00:00",
        "2021-06-29 08:00:00",
        "2021-07-19 11:00:00",
    ]


def test_test_period_positive_count(s01):
    """테스트 구간(09-01~09-14) 양성 28건."""
    d = s01.df
    test = d.loc[d.index >= s01.TEST_START]
    assert len(test) == 336
    assert int(test["y_cls"].sum()) == 28


# ══════════════════════════════════════════════════════════════════════
# 사양 정정 3건 (plan §2 발견 6 · 기타)
# ══════════════════════════════════════════════════════════════════════
def test_labor_cost_is_shift_multiplier_not_money(s01):
    """`인건비 == 1.0` ⟺ 9≤h≤17. h==8은 항상 1.5. 주말 할증 없음."""
    raw = s01.df_raw
    # ERP 결측일(예외 18행) 제외하고 검증
    normal = raw[~raw["날짜"].isin([20210713, 20210715])]
    is_one = normal["인건비"] == 1.0
    in_day = normal["시간"].between(9, 17)
    assert (is_one == in_day).all(), "인건비 1.0 구간이 9~17시와 정확히 일치하지 않는다"
    assert (normal.loc[normal["시간"] == 8, "인건비"] == 1.5).all()


def test_labor_cost_has_no_weekend_premium(s01):
    """요일 분포가 균등 → 주말 할증이 없다 (L4 레버 비용모델의 근거)."""
    d = s01.df
    prem = d[d["인건비"] == 1.5]
    counts = prem.groupby(prem.index.dayofweek).size()
    # 토(5)·일(6)이 평일과 같은 수준
    assert counts.max() / counts.min() < 1.15


def test_factory_headcount_is_continuous_not_count(s01):
    """`공장인원`은 인원수가 아니라 0~48.39 연속 실수(정규화 투입 공수)."""
    raw = s01.df_raw
    s = raw["공장인원"].dropna()
    assert s.min() == 0
    assert s.max() == pytest.approx(48.39, abs=0.01)
    assert s.nunique() > 3000, "정수 카운트라면 고유값이 이렇게 많을 수 없다"
    assert not (s == s.round()).all()


def test_seasonal_tariff_values(s01):
    """전기요금(계절): 1~2월 109.8 / 3~5·9월 167.2 / 6~8월 191.6."""
    d = s01.df
    by_month = d.groupby(d.index.month)["전기요금(계절)"].first()
    assert by_month.loc[[1, 2]].tolist() == [109.8, 109.8]
    assert by_month.loc[[3, 4, 5, 9]].tolist() == [167.2] * 4
    assert by_month.loc[[6, 7, 8]].tolist() == [191.6] * 3


# ══════════════════════════════════════════════════════════════════════
# 산출물
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "fid",
    ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11", "F12"],
)
def test_chapter1_figures_exist(s01, project_root, fid):
    """1장 그림 F01~F12가 모두 생성되었다."""
    hits = list((project_root / "outputs" / "figures").glob(f"{fid}_*.png"))
    assert hits, f"{fid} 그림이 없다"
    assert hits[0].stat().st_size > 5000, f"{fid} 그림이 비어 있다"


def test_figure_index_has_chapter1_rows(s01, project_root):
    """figure_index.csv 에 1장 그림이 등록되어 있다."""
    idx = pd.read_csv(project_root / "outputs" / "figures" / "figure_index.csv", encoding="utf-8-sig")
    ch1 = idx[idx["보고서절"].astype(str).str.startswith("1.")]
    assert len(ch1) >= 12
    assert ch1["fid"].is_unique
