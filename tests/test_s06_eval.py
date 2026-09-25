"""6장 — 성능평가 및 최종모델 선정 테스트.

FAST 모드에서는 절대 성능값이 달라지므로 **구조와 논리 일관성**을 검증한다.
특히 보고서 서술이 실측과 어긋나지 않는지(6.7절 교정)를 확인한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 6.1 회귀 성능표
# ══════════════════════════════════════════════════════════════════════
def test_regression_table_excludes_classifier(s06):
    """Task A 표에 회귀 예측이 없는 '피크 직접분류'가 들어가면 안 된다."""
    assert "피크 직접분류" not in set(s06.regression_tbl["모델"])
    assert "피크 직경분류" not in set(s06.regression_tbl["모델"])
    assert s06.regression_tbl["MAE"].notna().all()


def test_regression_table_has_required_columns(s06):
    """보고서 2.6절 표에 필요한 컬럼이 모두 있다."""
    cols = set(s06.regression_tbl.columns)
    for c in ("모델", "MAE", "RMSE", "sMAPE", "Peak-MAE", "학습시간(초)", "OOF MAE"):
        assert c in cols, c


def test_baseline_candidate_is_from_allowed_set(s06):
    """개선율의 분모는 확정된 베이스라인 후보 집합에서 나온다."""
    assert s06.BEST_BASELINE_NAME in s06.BASELINE_CANDIDATES


def test_no_masked_rows_in_test_period(s06):
    """9월 테스트 구간에는 계측정지·ERP결측이 없다 → 336행 그대로 쓴다."""
    assert s06.N_MASKED_IN_TEST == 0


def test_significance_uses_day_block_bootstrap(s06):
    """유의성 표에 CI와 p-value가 있고, 유의 판정이 CI와 일관된다."""
    t = s06.significance_tbl
    assert {"차이 95% CI", "p-value", "유의", "ci_low_raw", "ci_high_raw"} <= set(t.columns)
    # 표시용 문자열은 3자리로 반올림되어 0 근방에서 모호해지므로
    # 판정은 원시 경계값으로 검증한다.
    for _, r in t.iterrows():
        lo, hi = float(r["ci_low_raw"]), float(r["ci_high_raw"])
        expected = (lo > 0) or (hi < 0)
        assert bool(r["유의"]) == expected, f"{r['모델']}: CI [{lo},{hi}] vs 유의={r['유의']}"
        assert lo <= hi


# ══════════════════════════════════════════════════════════════════════
# 6.2 피크 탐지표
# ══════════════════════════════════════════════════════════════════════
def test_peak_table_includes_calendar_rule_row(s06):
    """2.7절 표 최상단에 달력규칙 기준 행이 있다."""
    assert (s06.peak_tbl["구분"] == "기준(달력규칙)").any()
    row = s06.peak_tbl[s06.peak_tbl["구분"] == "기준(달력규칙)"].iloc[0]
    assert row["Recall"] == 1.0
    assert int(row["FP"]) == s06.CALENDAR_RULE_FP


def test_peak_table_exposes_tau_per_row(s06):
    """각 행이 쓴 τ가 컬럼으로 노출된다 (plan 요구)."""
    assert "τ" in s06.peak_tbl.columns
    model_rows = s06.peak_tbl[s06.peak_tbl["구분"] != "기준(달력규칙)"]
    assert (model_rows["τ"] != "-").all()


def test_peak_table_confusion_sums_to_336(s06):
    """모든 행의 TP+FP+FN+TN이 336이다."""
    t = s06.peak_tbl
    assert ((t["TP"] + t["FP"] + t["FN"] + t["TN"]) == 336).all()


def test_fp_reduction_column_present(s06):
    """달력규칙 대비 FP 감소량이 정량화된다 — 제안모델의 실제 기여."""
    assert "FP 감소(달력규칙 대비)" in s06.peak_tbl.columns
    final = s06.peak_tbl[s06.peak_tbl["모델"] == s06.FINAL_MODEL_NAME].iloc[0]
    assert int(final["FP 감소(달력규칙 대비)"]) > 0


def test_oof_peak_table_has_more_positives_than_test(s06):
    """OOF 표가 비교의 주 근거다 — 양성이 테스트보다 훨씬 많다."""
    t = s06.peak_oof_tbl
    assert {"lift", "PR-AUC", "양성률"} <= set(t.columns)
    total_pos = t.iloc[0]["TP"] + t.iloc[0]["FN"]
    assert total_pos > 100, f"OOF 양성 {total_pos}건"


def test_lift_above_one_for_all_models(s06):
    """모든 모델의 lift(PR-AUC/양성률)가 1을 넘는다 = 무기술선보다 낫다."""
    assert (s06.peak_oof_tbl["lift"] > 1.0).all()


# ══════════════════════════════════════════════════════════════════════
# 6.3 Ablation
# ══════════════════════════════════════════════════════════════════════
def test_ablation_has_five_rows(s06):
    """보고서 2.8절 표의 5개 피처군이 모두 있다."""
    assert len(s06.ablation_tbl) == 5
    assert set(s06.ablation_tbl["제거 피처군"]) == {
        "과거 전력 지연변수", "생산량·공장인원", "공정상태 변수", "기상정보", "레짐분리"
    }


def test_production_ablation_replaces_regime_gate(s06):
    """'생산량 제외' 행에서 레짐 게이트도 생산량 비의존으로 교체된다.

    게이트가 생산량에 의존하면 생산량을 제거해도 정보가 새어 들어와
    보고서 1.2·2.3절이 약속한 검증이 성립하지 않는다.
    """
    row = s06.ablation_tbl[s06.ablation_tbl["제거 피처군"] == "생산량·공장인원"].iloc[0]
    if "레짐" in s06.FINAL_MODEL_NAME:
        assert "게이트도 생산량 비의존" in row["해석"]


def test_regime_separation_is_largest_contributor(s06):
    """레짐분리 제거가 MAE를 가장 크게 악화시킨다 = 기여 1위."""
    t = s06.ablation_tbl.set_index("제거 피처군")
    if "레짐" in s06.FINAL_MODEL_NAME:
        assert t.loc["레짐분리", "MAE 변화"] == t["MAE 변화"].max()
        assert t.loc["레짐분리", "MAE 변화"] > 0


# ══════════════════════════════════════════════════════════════════════
# 6.4 스코어카드 · 추론시간 · 대리모델
# ══════════════════════════════════════════════════════════════════════
def test_scorecard_covers_five_criteria(s06):
    """보고서 2.9절 5개 선정기준이 모두 점수화된다."""
    cols = " ".join(s06.scorecard.columns)
    for k in ("1)", "2)", "3)", "4)", "5)"):
        assert k in cols, k
    assert "종합점수" in s06.scorecard.columns


def test_final_model_is_scorecard_winner(s06):
    """최종모델 = 스코어카드 1위 (MAE 최소 모델이 아닐 수 있다)."""
    assert s06.FINAL_MODEL_NAME == s06.scorecard.iloc[0]["모델"]
    assert s06.scorecard["종합점수"].is_monotonic_decreasing


def test_inference_time_is_predict_only(s06):
    """추론시간은 학습을 포함하지 않는다 — 학습시간보다 짧아야 한다."""
    fit_sec = float(s06.cv_results[s06.FINAL_MODEL_NAME]["fit_sec"])
    assert s06.INFER_SEC > 0
    assert s06.INFER_SEC < max(fit_sec, 1.0) * 5, (
        f"추론 {s06.INFER_SEC}초 vs 학습 {fit_sec}초 — 학습이 섞인 것 같다"
    )


def test_surrogate_rule_decided_with_fidelity(s06):
    """설명용 대리모델 사용 여부와 충실도가 확정된다 (7장이 추측하지 않도록)."""
    assert isinstance(s06.NEEDS_SURROGATE, bool)
    assert 0.0 <= s06.SURROGATE_FIDELITY <= 1.0
    if s06.NEEDS_SURROGATE:
        assert s06.SURROGATE_NAME != s06.FINAL_MODEL_NAME
        assert s06.SURROGATE_FIDELITY > 0.7, "대리 충실도가 낮으면 설명을 신뢰할 수 없다"
    else:
        assert s06.SURROGATE_FIDELITY == 1.0


def test_improvement_percentages_consistent(s06):
    """개선율이 MAE 값으로부터 실제로 계산된 값과 일치한다."""
    expected = (s06.BEST_BASELINE_MAE - s06.FINAL_MAE) / s06.BEST_BASELINE_MAE * 100
    assert s06.IMPROVE_MAE_PCT == pytest.approx(expected, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
# 6.5 D2 강건성
# ══════════════════════════════════════════════════════════════════════
def test_d2_rerun_covers_all_regression_models(s06):
    """D1/D2 대비표가 회귀 모델 전부를 담는다."""
    assert set(s06.d1_vs_d2["모델"]) == set(s06.REGRESSION_MODELS)


def test_d2_has_fewer_samples(s06):
    """D2는 복제 제거로 표본이 줄어든다 → n 병기가 필수."""
    assert {"D1 학습 n", "D2 학습 n"} <= set(s06.d1_vs_d2.columns)
    assert (s06.d1_vs_d2["D2 학습 n"] < s06.d1_vs_d2["D1 학습 n"]).all()


def test_rank_correlation_is_high(s06):
    """D1↔D2 순위상관이 높다 = 결론이 복제 처리 방식에 강건하다."""
    rho = float(s06.rank_preservation.iloc[0]["값"])
    assert rho > 0.7


def test_regime_family_stays_on_top_in_d2(s06):
    """D2에서도 레짐 계열이 상위를 지킨다 (구조적 선택의 강건성)."""
    top2 = s06.d1_vs_d2.nsmallest(2, "D2 OOF MAE")["모델"].tolist()
    assert any("레짐" in m for m in top2)


# ══════════════════════════════════════════════════════════════════════
# 6.6 변수 제거 결과 검증 — 지연변수 제거 효과의 분해
# ══════════════════════════════════════════════════════════════════════
def test_frame_argument_defaults_to_feat(s06):
    """frame 인자를 주지 않으면 기존과 같은 데이터가 나온다(추가형 변경)."""
    a = s06.get_fold_data(3, "D1")
    b = s06.get_fold_data(3, "D1", frame=s06.feat)
    for x, y in zip(a[:4], b[:4]):
        pd.testing.assert_frame_equal(x, y)


def test_lag_variants_cover_the_plan(s06):
    """피처군 6개, 지연 세분 3개, 조합, 휴무 인지형 2종, 잡음 기준선 5회가 모두 있다."""
    ids = set(s06.lag_variant_tbl["ID"])
    assert {"A", "L24", "L48", "L168", "B+W", "SA-all", "SA-168"} <= ids
    assert sum(i.startswith("G-") for i in ids) == len(s06.FEATURE_GROUPS)
    assert sum(i.startswith("NULL-") for i in ids) == len(s06.LAG_NOISE_SEEDS)


def test_lag_base_is_final_model_oof(s06):
    """기준 행은 최종모델 교차검증 결과를 그대로 쓴다(재학습 없음)."""
    a = s06.lag_variant_tbl.set_index("ID").loc["A"]
    oof = s06.cv_results[s06.FINAL_MODEL_NAME]["oof"]
    assert abs(a["OOF MAE"] - s06.mae(oof["y_avg"], oof["pred_avg"])) < 1e-12


def test_lag_group_removal_matches_ablation(s06):
    """6.6절 피처군 제거는 6.3절 Ablation 과 같은 값을 낸다(같은 코드 경로)."""
    ref = s06.lag_variant_tbl.set_index("ID")
    abl = s06.ablation_tbl.set_index("제거 피처군")
    pairs = {"G-과거전력": "과거 전력 지연변수", "G-생산정보": "생산량·공장인원",
             "G-공정상태": "공정상태 변수", "G-기상정보": "기상정보"}
    for vid, label in pairs.items():
        assert abs(ref.loc[vid, "OOF MAE"] - abl.loc[label, "MAE"]) < 6e-4, vid


def test_status_matched_lag_references_past_same_status_day(s06):
    """휴무 인지형 지연값은 원점 이전의, 대상일과 같은 가동 상태인 날의 같은 시각 값이다."""
    f = s06.feat
    st = s06.operating_calendar["is_shutdown"]
    vals, walked = s06.status_matched_lag(f, "y_avg", 24, 1, 14)
    moved = np.where(walked > 0)[0]
    assert len(moved) > 0
    for i in moved[:: max(1, len(moved) // 150)]:
        t = f.index[i]
        ref = t - pd.Timedelta(hours=24) - pd.Timedelta(days=int(walked[i]))
        assert ref < t.normalize()
        assert bool(st[ref.normalize()]) == bool(st[t.normalize()])
        assert np.isclose(vals[i], f["y_avg"].get(ref), equal_nan=True)


def test_lag_criteria_and_noise_band(s06):
    """잡음 폭은 양수이고, 판단 기준 열이 모두 있다."""
    assert s06.LAG_NOISE_MAE > 0 and s06.LAG_NOISE_PEAK >= 0
    cols = set(s06.lag_variant_tbl.columns)
    for c in ("① CI가 0 미포함", "② 정상 fold 악화 ≤ 잡음", "③ 정상일 악화 ≤ 잡음", "④ Recall·Peak-MAE", "①~④ 통과"):
        assert c in cols, c


def test_lag_decomposition_and_top_days(s06):
    """보고서 표 2-6 행과 오차 상위 2일이 만들어진다."""
    assert len(s06.lag_decomp_tbl) == len(s06.LAG_DECOMP_IDS)
    assert len(s06.LAG_TOP_DAYS) == 2
    assert 0 < s06.LAG_TOP_SHARE < 1


def test_lag_d2_starts_with_base(s06):
    d2 = s06.lag_d2_tbl
    assert d2.iloc[0]["ID"] == "A" and set(d2["조건"]) == {"D2"}
    assert set(s06.LAG_D2_FIXED) <= set(d2["ID"])


def test_lofo_only_in_full_mode(s06):
    """개별 변수 제거 44회는 본실행에서만 돈다."""
    expected = 0 if s06.FAST else len(s06.FEATURE_COLS)
    assert len(s06.lofo_tbl) == expected


def test_gate_diagnosis_for_regime_model(s06):
    """레짐 모델이면 오차 상위 4일의 게이트 배정이 기록된다."""
    if "레짐" not in s06.FINAL_MODEL_NAME:
        pytest.skip("레짐 모델이 아님")
    g = s06.gate_diag_tbl
    assert len(g) == 4
    assert (g["게이트 휴무일 변수 분기 수"] >= 0).all()
    assert g["08~17시 배정 레짐"].str.len().gt(0).all()


# ══════════════════════════════════════════════════════════════════════
# 6.7 결과 해석 — 보고서 서술 교정
# ══════════════════════════════════════════════════════════════════════
def test_interpretation_table_exists(s06):
    """실측과 어긋나는 서술을 교정하는 표가 생성된다."""
    assert len(s06.interpretation_tbl) >= 3
    assert {"항목", "실측", "보고서 서술 교정"} <= set(s06.interpretation_tbl.columns)


def test_significance_narrative_matches_evidence(s06):
    """유의하지 않으면 '유의하지 않다'고 서술한다 (과대 주장 방지)."""
    row = s06.interpretation_tbl[
        s06.interpretation_tbl["항목"] == "MAE 개선의 유의성"
    ].iloc[0]
    if s06.MAE_IMPROVEMENT_SIGNIFICANT:
        assert "유의하다" in row["보고서 서술 교정"]
    else:
        assert "유의하지 않다" in row["보고서 서술 교정"]
        # 대안 근거를 반드시 제시해야 한다
        assert "Peak-MAE" in row["보고서 서술 교정"]


def test_lag_ablation_narrative_matches_sign(s06):
    """지연변수 제거 효과의 부호에 맞는 서술이 생성된다."""
    row = s06.interpretation_tbl[
        s06.interpretation_tbl["항목"] == "과거전력 지연변수 Ablation"
    ]
    if len(row):
        d = float(
            s06.ablation_tbl[
                s06.ablation_tbl["제거 피처군"] == "과거 전력 지연변수"
            ]["MAE 변화"].iloc[0]
        )
        text = row.iloc[0]["보고서 서술 교정"]
        if d < 0:
            assert "개선" in text and "휴무" in text
        else:
            assert "악화" in text


@pytest.mark.parametrize("fid", ["F17", "F18", "F19", "F20", "F23", "F24", "F25"])
def test_chapter6_figures_exist(s06, project_root, fid):
    hits = list((project_root / "outputs" / "figures").glob(f"{fid}_*.png"))
    assert hits and hits[0].stat().st_size > 5000
