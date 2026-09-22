"""6장 — 성능평가 및 최종모델 선정 테스트.

FAST 모드에서는 절대 성능값이 달라지므로 **구조와 논리 일관성**을 검증한다.
특히 보고서 서술이 실측과 어긋나지 않는지(6.6절 교정)를 확인한다.
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
# 6.6 결과 해석 — 보고서 서술 교정
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
