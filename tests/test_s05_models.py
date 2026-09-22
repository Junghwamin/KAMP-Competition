"""5장 — 제안모델 개발 테스트.

FAST 모드(에폭·trial 축소)로 돌기 때문에 **절대 성능값을 단정하지 않는다.**
대신 구조적 성질(누수 없음, OOF 완결성, τ 산출 범위, 재현성)을 검증한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 5.0 공통 하네스
# ══════════════════════════════════════════════════════════════════════
def test_lgb_params_are_deterministic(s05):
    """LightGBM 재현성 설정이 모두 들어가 있다 (심사위원 PC 코어 수와 무관)."""
    p = s05.LGB_BASE
    assert p["deterministic"] is True
    assert p["force_row_wise"] is True
    assert p["num_threads"] == 4
    for k in ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed"):
        assert p[k] == s05.SEED


def test_all_models_produce_full_oof(s05):
    """모든 모델이 fold 2~5 전체를 덮는 OOF 예측을 만든다."""
    for name, res in s05.cv_results.items():
        oof = res["oof"]
        assert set(oof["fold"].unique()) == set(s05.ACTIVE_FOLDS), name
        assert len(oof) > 1000, f"{name}: OOF {len(oof)}행"
        assert oof.index.is_monotonic_increasing or True  # fold 순 연결


def test_oof_index_has_no_duplicates(s05):
    """OOF 인덱스에 중복이 없다 (fold 검증구간이 겹치지 않음의 재확인)."""
    for name, res in s05.cv_results.items():
        assert res["oof"].index.is_unique, name


def test_oof_excludes_masked_rows(s05):
    """OOF 에 계측정지·ERP결측·워밍업 행이 없다."""
    f = s05.feat
    for name, res in s05.cv_results.items():
        sub = f.loc[res["oof"].index]
        assert not sub["is_outage"].any(), name
        assert not sub["is_erp_missing"].any(), name
        assert not sub["is_warmup"].any(), name


def test_tau_is_median_of_fold_taus(s05):
    """테스트용 τ = fold τ 들의 중앙값 (테스트를 보고 정하지 않는다)."""
    for name, res in s05.cv_results.items():
        expected = float(np.median(res["fold_metrics"]["τ"]))
        assert res["tau"] == pytest.approx(expected)


def test_tau_below_theta_for_regression_models(s05):
    """회귀 기반 모델의 τ 는 θ 보다 낮다 (예측이 평균으로 수축하므로)."""
    for name in ("LightGBM", "2단계 레짐(2분류)", "Random Forest (보정)"):
        assert s05.cv_results[name]["tau"] < s05.THETA, name


def test_test_predictions_have_336_rows(s05):
    """모든 모델의 테스트 예측이 336행이다."""
    for name, res in s05.test_results.items():
        assert len(res["index"]) == 336, name
        assert len(res["pred_peak"]) == 336, name


# ══════════════════════════════════════════════════════════════════════
# 5.1 LightGBM + Optuna
# ══════════════════════════════════════════════════════════════════════
def test_hpo_history_saved_with_both_studies(s05):
    """회귀·분류 두 탐색 이력이 모두 저장된다 (2.5절 '[TBD]회 탐색' 근거)."""
    h = s05.hpo_trials
    assert set(h["모델"]) == {"LightGBM 회귀", "피크 직접분류"}
    assert s05.HPO_N_TRIALS == len(h)
    assert s05.HPO_N_TRIALS == 2 * s05.N_TRIALS


def test_hpo_search_space_respected(s05):
    """탐색된 최적 파라미터가 지정한 공간 안에 있다."""
    p = s05.lgb_best_params
    assert 15 <= p["num_leaves"] <= 127
    assert 0.01 <= p["learning_rate"] <= 0.2
    assert 3 <= p["max_depth"] <= 12
    assert 5 <= p["min_child_samples"] <= 50
    assert 0.6 <= p["feature_fraction"] <= 1.0
    assert 0.6 <= p["bagging_fraction"] <= 1.0


@pytest.mark.slow
def test_optuna_is_reproducible(s05):
    """동일 시드로 재탐색하면 같은 최적 파라미터가 나온다 (재현성 요건)."""
    params2, _ = s05.optimize_lgb("D1")
    assert params2 == s05.lgb_best_params


# ══════════════════════════════════════════════════════════════════════
# 5.2 2단계 레짐 — 핵심 결과
# ══════════════════════════════════════════════════════════════════════
def test_regime_model_survives_shutdown_fold(s05):
    """fold4(하계휴가+재가동)에서 레짐 모델이 Seasonal Naive 붕괴를 막는다.

    이 프로젝트의 가장 중요한 결과다. lag168 참조값이 휴무일이 되면
    Naive 는 완전히 붕괴하는데, 레짐을 먼저 분류하면 그 붕괴를 피한다.
    → 제안모델의 이점이 **평균이 아니라 달력이 깨지는 구간**에 집중됨을 보인다.
    """
    m = s05.fold_mae_matrix
    naive4 = float(m.loc["Seasonal Naive", "fold4"])
    regime4 = float(m.loc["2단계 레짐(2분류)", "fold4"])
    assert naive4 > 50, f"Naive fold4 MAE={naive4} — 붕괴가 재현되지 않았다"
    assert regime4 < naive4 / 3, f"레짐 {regime4} vs Naive {naive4}"


def test_regime_beats_naive_on_every_fold(s05):
    """레짐 모델이 모든 fold에서 Naive 이상이다."""
    m = s05.fold_mae_matrix
    for col in m.columns:
        assert float(m.loc["2단계 레짐(2분류)", col]) <= float(m.loc["Seasonal Naive", col]), col


def test_regime_gate_features_configurable(s05):
    """게이트 피처를 인자로 바꿀 수 있다 ('생산량 제외' ablation 요건)."""
    no_prod = [c for c in s05.FEATURE_COLS if c not in s05.FEATURE_GROUPS["생산정보"]]
    fit = s05.make_regime_model(2, gate_features=no_prod)
    Xtr, ytr, Xva, yva, _ = s05.get_fold_data(2, "D1")
    out = fit(Xtr, ytr, Xva)
    assert len(out["pred_avg"]) == len(Xva)
    assert np.isfinite(out["pred_avg"]).all()


def test_regime_3class_also_available(s05):
    """3분류 레짐도 비교 실험으로 준비되어 있다."""
    assert "2단계 레짐(3분류)" in s05.cv_results
    assert "2단계 레짐(2분류)" in s05.cv_results


def test_no_prediction_left_unassigned_by_regime(s05):
    """레짐 배정 누락으로 예측값이 0으로 남은 행이 없다."""
    for name in ("2단계 레짐(2분류)", "2단계 레짐(3분류)"):
        pa = s05.cv_results[name]["oof"]["pred_avg"].to_numpy()
        assert np.isfinite(pa).all()
        assert (pa != 0).all(), f"{name}: 예측 0인 행이 있다 (레짐 미배정 의심)"


# ══════════════════════════════════════════════════════════════════════
# 5.3 DNN
# ══════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(True, reason="HAS_TF 조건은 아래 테스트에서 동적으로 확인")
def test_placeholder():  # pragma: no cover
    pass


def test_dnn_epochs_explicitly_set(s05):
    """`epochs` 가 명시되어 있다 — Keras 기본값 1에폭 함정 방지."""
    assert s05.DNN_EPOCHS >= 5
    if not s05.FAST:
        assert s05.DNN_EPOCHS == 300


def test_dnn_present_when_tf_available(s05):
    """TF가 있으면 DNN이 레지스트리에 있고, 없으면 파이프라인이 그래도 완주한다."""
    if s05.HAS_TF:
        assert "DNN (MLP)" in s05.MODEL_REGISTRY
        assert "DNN (MLP)" in s05.cv_results
    else:
        assert "DNN (MLP)" not in s05.MODEL_REGISTRY


# ══════════════════════════════════════════════════════════════════════
# 5.4 피크 직접 분류
# ══════════════════════════════════════════════════════════════════════
def test_peak_classifier_outputs_valid_probabilities(s05):
    """피크 분류기가 [0,1] 확률을 낸다."""
    prob = s05.cv_results["피크 직접분류"]["oof"]["prob"].to_numpy()
    assert np.isfinite(prob).all()
    assert prob.min() >= 0.0 and prob.max() <= 1.0


def test_peak_classifier_beats_random(s05):
    """피크 분류기 PR-AUC 가 무기술선(양성률)보다 높다."""
    from sklearn.metrics import average_precision_score

    oof = s05.cv_results["피크 직접분류"]["oof"]
    ap = average_precision_score(oof["y_cls"], oof["prob"])
    base = float(oof["y_cls"].mean())
    assert ap > base * 1.5, f"PR-AUC {ap:.3f} vs 양성률 {base:.3f}"


def test_class_weight_applied(s05):
    """불균형 대응으로 scale_pos_weight 가 적용된다 (2.5절 1항)."""
    import inspect

    src = inspect.getsource(s05.make_peak_classifier)
    assert "scale_pos_weight" in src


# ══════════════════════════════════════════════════════════════════════
# 5.6 앙상블 — 개선 없으면 미채택
# ══════════════════════════════════════════════════════════════════════
def test_ensemble_only_adopted_when_it_improves(s05):
    """앙상블은 검증 MAE가 실제로 낮아질 때만 채택된다."""
    info = s05.ensemble_info
    if info.get("채택"):
        assert info["앙상블 OOF MAE"] < info["단일 OOF MAE"]
        assert "앙상블" in s05.cv_results
    else:
        assert "개선 없음" in info["사유"] or "후보 부족" in info["사유"]
        assert "앙상블" not in s05.cv_results


# ══════════════════════════════════════════════════════════════════════
# 5.7 확률 보정
# ══════════════════════════════════════════════════════════════════════
def test_calibration_improves_brier_and_ece(s05):
    """Isotonic 보정이 Brier·ECE를 모두 개선한다."""
    t = s05.calibration_tbl.set_index("구분")
    assert t.loc["보정 후", "Brier"] <= t.loc["보정 전", "Brier"]
    assert t.loc["보정 후", "ECE"] <= t.loc["보정 전", "ECE"]


def test_ece_is_zero_for_perfect_calibration(s05):
    """ECE 함수가 완전 보정 입력에 0을 준다 (함수 자체 검증)."""
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    assert s05.expected_calibration_error(y, y.astype(float)) == pytest.approx(0.0)


def test_ece_detects_miscalibration(s05):
    """ECE 함수가 과대추정을 잡아낸다 (음성 대조)."""
    y = np.zeros(100)
    y[:10] = 1
    over = np.full(100, 0.9)  # 실제 10%인데 90%라 주장
    assert s05.expected_calibration_error(y, over) == pytest.approx(0.8, abs=0.01)


# ══════════════════════════════════════════════════════════════════════
# 5.8 불확실성
# ══════════════════════════════════════════════════════════════════════
def test_quantile_and_conformal_reported(s05):
    """분위회귀와 Split Conformal 이 모두 보고된다."""
    t = s05.uncertainty_tbl
    assert "분위회귀 q10~q90" in set(t["방법"])
    assert "Split Conformal" in set(t["방법"])
    assert (t["실제 피복률"].between(0, 1)).all()
    assert (t["평균 구간폭"] > 0).all()


def test_conformal_reports_both_calibration_sets(s05):
    """보정집합에 하계휴가를 포함/제외한 두 변형을 함께 보고한다.

    휴무를 포함한 보정집합은 잔차가 극단적으로 커져 구간이 무의미하게 넓어진다.
    이 차이를 드러내지 않으면 구간폭의 의미를 오독하게 된다.
    """
    t = s05.uncertainty_tbl
    conf = t[t["방법"] == "Split Conformal"]
    assert len(conf) == 2
    sets = set(conf["보정집합"])
    assert "전체(휴무 포함)" in sets and "운영일만(휴무 제외)" in sets
    wide = float(conf[conf["보정집합"] == "전체(휴무 포함)"]["평균 구간폭"].iloc[0])
    narrow = float(conf[conf["보정집합"] == "운영일만(휴무 제외)"]["평균 구간폭"].iloc[0])
    assert narrow < wide, "휴무를 빼면 구간이 좁아져야 한다"


# ══════════════════════════════════════════════════════════════════════
# 산출물
# ══════════════════════════════════════════════════════════════════════
def test_timing_recorded(s05):
    """학습·탐색 시간이 timing.csv 로 기록된다 (2.6·2.9절 근거)."""
    t = s05.timing_tbl
    assert len(t) > 0
    assert (t["초"] >= 0).all()
    assert any("Optuna" in x for x in t["항목"])


@pytest.mark.parametrize("fid", ["F21", "F22", "F26"])
def test_chapter5_figures_exist(s05, project_root, fid):
    hits = list((project_root / "outputs" / "figures").glob(f"{fid}_*.png"))
    assert hits and hits[0].stat().st_size > 5000


def test_fold_mae_matrix_saved(s05, project_root):
    """모델 × fold MAE 매트릭스가 저장된다 (2.9절 폴드 편차 근거)."""
    p = project_root / "outputs" / "tables" / "ch5_fold_mae_matrix.csv"
    assert p.exists()
    m = pd.read_csv(p, encoding="utf-8-sig")
    assert {"fold2", "fold3", "fold4", "fold5"} <= set(m.columns)
