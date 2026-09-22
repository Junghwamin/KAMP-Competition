"""3장 — 분할 설계 및 평가지표 테스트.

이 장에서 3~7장이 공유하는 계약이 확정되므로, 계약이 깨지면 이후 모든 장이
무의미해진다. 오라클은 plan.md §6 및 프로젝트 메모리에서 온다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 3.1 데이터 조건 2축
# ══════════════════════════════════════════════════════════════════════
def test_data_conditions_day_counts(s03):
    """D1 = 257일(원형), D2 = 142일(복제 제거)."""
    t = s03.cond_tbl.set_index("조건")
    assert int(t.loc["D1", "일수"]) == 257
    assert int(t.loc["D2", "일수"]) == 142
    assert int(t.loc["D1", "시간수"]) == 6168


def test_dedup_keeps_one_day_per_group(s03):
    """D2는 각 프로파일 중복그룹에서 정확히 1일만 남긴다."""
    keep = s03.DEDUP_KEEP_DAYS
    assert len(keep) == 142
    kept_hashes = [s03.profile_hashes.loc[d] for d in keep]
    assert len(set(kept_hashes)) == 142, "잔류일들의 프로파일이 서로 달라야 한다"


def test_dedup_selection_is_deterministic(s03):
    """D2 잔류일 선택이 결정론적(시간순 최초)이다 — 재현성 요건."""
    again = s03.build_dedup_day_set(s03.profile_hashes)
    assert again == s03.DEDUP_KEEP_DAYS


# ══════════════════════════════════════════════════════════════════════
# 3.2 fold 구성 + 복제 오염
# ══════════════════════════════════════════════════════════════════════
def test_fold1_is_contaminated_and_dropped(s03):
    """fold1은 검증 14일 중 7일(50%)이 학습구간과 동일 프로파일 → 폐기."""
    c = s03.contamination.set_index("fold")
    assert int(c.loc[1, "오염일수"]) == 7
    assert float(c.loc[1, "오염률"]) == 50.0
    assert 1 in s03.DROPPED_FOLDS
    assert s03.ACTIVE_FOLDS == (2, 3, 4, 5)


def test_active_folds_and_test_are_clean(s03):
    """사용 fold(2~5)와 테스트 구간의 복제 오염이 0이다."""
    c = s03.contamination.set_index("fold")
    for f in s03.ACTIVE_FOLDS:
        assert int(c.loc[f, "오염일수"]) == 0
    assert int(c.loc["TEST", "오염일수"]) == 0


def test_gate3_passes(s03):
    """게이트 3 전 항목 통과."""
    assert s03.gate3["통과"].all()


def test_fold_train_q95_never_equals_theta(s03):
    """fold별 학습구간 q95 = 182/182/184/186/186 — 어디서도 187이 안 나온다.

    θ를 fold마다 재산출하면 y_cls 정의 자체가 달라져 fold 간 Recall 평균이
    무의미해진다. **θ 고정이 필수**임을 보이는 핵심 증거.
    """
    q = s03.folds_tbl["학습구간 q95"].tolist()
    assert q == [182.0, 182.0, 184.0, 186.0, 186.0]
    assert s03.THETA not in q


def test_fold_gap_is_24_hours(s03):
    """학습 종료와 검증 시작 사이에 정확히 24시간 갭이 있다."""
    for fold, tr_end, vs, _ in s03.FOLD_SPEC:
        tr, va = s03.fold_slices(s03.df, fold)
        last_train = s03.df.index[tr].max()
        first_val = s03.df.index[va].min()
        assert (first_val - last_train) == pd.Timedelta(hours=25), (
            f"fold{fold}: {last_train} → {first_val}"
        )


def test_folds_are_expanding_and_ordered(s03):
    """학습창이 fold마다 확장되고, 검증구간이 시간순으로 이동한다."""
    sizes, starts = [], []
    for fold, *_ in s03.FOLD_SPEC:
        tr, va = s03.fold_slices(s03.df, fold)
        sizes.append(int(tr.sum()))
        starts.append(s03.df.index[va].min())
    assert sizes == sorted(sizes), "학습창이 확장되지 않는다"
    assert starts == sorted(starts), "검증구간이 시간순이 아니다"


def test_no_train_val_overlap(s03):
    """어떤 fold도 학습·검증이 겹치지 않는다."""
    for fold, *_ in s03.FOLD_SPEC:
        tr, va = s03.fold_slices(s03.df, fold)
        assert not (tr & va).any()


def test_validation_window_is_14_days(s03):
    """검증구간은 fold마다 14일(336시간)이다."""
    for fold, *_ in s03.FOLD_SPEC:
        _, va = s03.fold_slices(s03.df, fold)
        assert int(va.sum()) == 336


def test_fold_shutdown_ratio_recorded(s03):
    """fold별 휴무 비중이 기록된다 — 성능 편차 오독 방지용."""
    f = s03.folds_tbl.set_index("fold")
    ratios = {int(k): float(v.strip("%")) for k, v in f["휴무비중"].items()}
    # 하계휴가가 걸린 fold3·fold4의 휴무 비중이 가장 높아야 한다
    assert ratios[3] > ratios[2]
    assert ratios[4] > ratios[5]
    assert ratios[4] >= 40


# ══════════════════════════════════════════════════════════════════════
# 3.3 지표 함수
# ══════════════════════════════════════════════════════════════════════
def test_mae_rmse_basic(s03):
    y, p = np.array([1.0, 2.0, 3.0]), np.array([1.0, 4.0, 3.0])
    assert s03.mae(y, p) == pytest.approx(2 / 3)
    assert s03.rmse(y, p) == pytest.approx(np.sqrt(4 / 3))


def test_smape_excludes_zero_denominator(s03):
    """분모가 0인 시각(y=ŷ=0)은 sMAPE 계산에서 제외된다."""
    y = np.array([0.0, 10.0])
    p = np.array([0.0, 20.0])
    # 두 번째 점만: |10-20| / 15 = 0.6667 → 66.67%
    assert s03.smape(y, p) == pytest.approx(66.667, abs=0.01)


def test_peak_mae_only_counts_peak_hours(s03):
    """Peak-MAE 는 y >= θ 인 시각만 센다."""
    y = np.array([100.0, 200.0])
    p = np.array([150.0, 210.0])
    assert s03.peak_mae(y, p, 187) == pytest.approx(10.0)


def test_clopper_pearson_matches_scipy_reference(s03):
    """Clopper-Pearson 구간이 scipy `binomtest(...).proportion_ci('exact')` 와 일치한다."""
    from scipy.stats import binomtest

    for k, n in [(10, 28), (3, 40), (0, 12), (25, 25), (1, 200)]:
        ref = binomtest(k, n).proportion_ci(method="exact")
        lo, hi = s03.clopper_pearson(k, n)
        assert lo == pytest.approx(float(ref.low), abs=1e-9)
        assert hi == pytest.approx(float(ref.high), abs=1e-9)
    # 경계 케이스
    assert s03.clopper_pearson(0, 10)[0] == 0.0
    assert s03.clopper_pearson(10, 10)[1] == 1.0


def test_test_peak_ci_is_too_wide_to_discriminate(s03):
    """테스트 28건 기준 Recall CI 폭이 0.37 → 모델 구분 불가."""
    lo, hi = s03.clopper_pearson(10, 28)
    assert (hi - lo) == pytest.approx(0.37, abs=0.01)


def test_classification_metrics_confusion_counts(s03):
    """혼동행렬 4칸과 Precision/Recall/F1이 정확하다."""
    y = np.array([1, 1, 0, 0, 1])
    p = np.array([1, 0, 1, 0, 1])
    m = s03.classification_metrics(y, p)
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (2, 1, 1, 1)
    assert m["Recall"] == pytest.approx(2 / 3)
    assert m["Precision"] == pytest.approx(2 / 3)


def test_lift_normalizes_pr_auc_by_base_rate(s03):
    """lift = PR-AUC / 양성률 — fold별 양성률 차이(2.9배)를 정규화한다."""
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.1).astype(int)
    score = rng.random(400)
    m = s03.classification_metrics(y, (score > 0.5).astype(int), score)
    assert m["lift"] == pytest.approx(m["PR-AUC"] / m["양성률"])


def test_paired_bootstrap_detects_real_difference(s03):
    """명확히 더 나은 모델은 CI가 0을 포함하지 않는다."""
    idx = pd.date_range("2021-01-01", periods=240, freq="h")
    rng = np.random.default_rng(1)
    y = rng.normal(100, 10, 240)
    good = y + rng.normal(0, 1, 240)
    bad = y + rng.normal(0, 15, 240)
    r = s03.paired_bootstrap_mae(y, bad, good, index=idx)
    assert r["diff"] > 0          # bad 가 더 큰 오차
    assert r["유의"] is True
    assert r["ci_low"] > 0


def test_paired_bootstrap_no_false_positive(s03):
    """오차 크기가 동일한 두 모델은 유의하지 않다고 판정한다 (음성 대조).

    부호만 반대인 오차를 주어 `|오차|` 분포가 **정확히 동일**하게 만든다.
    서로 다른 난수로 뽑으면 유한표본에서 실제로 한쪽이 나을 수 있어
    음성 대조가 성립하지 않는다.
    """
    idx = pd.date_range("2021-01-01", periods=240, freq="h")
    rng = np.random.default_rng(2)
    y = rng.normal(100, 10, 240)
    eps = rng.normal(0, 5, 240)
    a, b = y + eps, y - eps  # |y-a| == |y-b| 이므로 MAE 차이는 정확히 0
    r = s03.paired_bootstrap_mae(y, a, b, index=idx)
    assert r["diff"] == pytest.approx(0.0, abs=1e-12)
    assert r["유의"] is False
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_paired_bootstrap_is_reproducible(s03):
    """동일 시드에서 부트스트랩 결과가 완전히 동일하다 (재현성 요건)."""
    idx = pd.date_range("2021-01-01", periods=120, freq="h")
    rng = np.random.default_rng(3)
    y = rng.normal(100, 10, 120)
    a, b = y + 1.0, y + 2.0
    r1 = s03.paired_bootstrap_mae(y, a, b, index=idx)
    r2 = s03.paired_bootstrap_mae(y, a, b, index=idx)
    assert r1 == r2


def test_tune_tau_beats_fixed_theta_on_shrunk_predictions(s03):
    """회귀 예측이 평균으로 수축할 때 τ 조정이 θ 고정보다 F1이 높다.

    plan.md 가 보고한 3.5배 차이의 메커니즘을 검증한다.
    """
    rng = np.random.default_rng(4)
    y_peak = rng.normal(150, 30, 1500)
    # 평균으로 수축한 예측 (회귀모델의 전형)
    pred = 150 + (y_peak - 150) * 0.45 + rng.normal(0, 4, 1500)
    theta = 187.0
    y_true = (y_peak >= theta).astype(int)

    m_theta = s03.classification_metrics(y_true, (pred >= theta).astype(int))
    tau = s03.tune_tau(y_peak, pred, theta)
    m_tau = s03.classification_metrics(y_true, (pred >= tau).astype(int))

    assert tau < theta, "수축한 예측에는 더 낮은 τ가 맞다"
    assert m_tau["F1"] > m_theta["F1"]
    assert m_tau["Recall"] > m_theta["Recall"]


# ══════════════════════════════════════════════════════════════════════
# 3.4 달력규칙 기준선
# ══════════════════════════════════════════════════════════════════════
def test_calendar_rule_exact_confusion_matrix(s03):
    """달력규칙 `평일 ∧ 08≤h≤18` → TP 28 / FN 0 / FP 82 / TN 226.

    Recall 1.000 이므로 **어떤 ML 모델도 Recall 로는 이길 수 없다**.
    Task B 주지표를 F1/PR-AUC 로 재정의해야 하는 근거.
    """
    m = s03.calendar_rule_metrics
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (28, 0, 82, 226)
    assert m["Recall"] == 1.0
    assert m["TP"] + m["FN"] + m["FP"] + m["TN"] == 336


def test_calendar_rule_fp_is_the_target_to_beat(s03):
    """제안모델이 줄여야 할 대상은 FP 82건이다."""
    assert s03.CALENDAR_RULE_FP == 82


# ══════════════════════════════════════════════════════════════════════
# 3.5 fold 데이터 헬퍼
# ══════════════════════════════════════════════════════════════════════
def test_get_test_data_has_exactly_336_rows(s03):
    """테스트는 마스크와 무관하게 336행을 유지한다 (제출 파일 요건)."""
    _, _, Xte, yte, idx = s03.get_test_data("D1")
    assert len(Xte) == 336 and len(yte) == 336 and len(idx) == 336
    assert not Xte.isna().any().any()


def test_masks_excluded_from_training(s03):
    """워밍업·계측정지·ERP결측이 학습에서 제외된다."""
    ok = s03.usable_mask(s03.feat, "D1")
    f = s03.feat
    assert not (ok & f["is_warmup"]).any()
    assert not (ok & f["is_outage"]).any()
    assert not (ok & f["is_erp_missing"]).any()


def test_fold2_validation_excludes_erp_missing_days(s03):
    """fold2 검증구간(07-07~07-20)에 든 7/13·7/15 48행이 제외된다."""
    _, _, Xva, _, idx = s03.get_fold_data(2, "D1")
    assert len(Xva) == 288, "336 - 48(ERP 결측) = 288 이어야 한다"
    days = {str(t.date()) for t in idx}
    assert "2021-07-13" not in days and "2021-07-15" not in days


def test_d2_reduces_training_samples(s03):
    """D2는 복제를 제거하므로 학습 표본이 크게 준다 (표에 n 병기 필요)."""
    for fold in s03.ACTIVE_FOLDS:
        n1 = len(s03.get_fold_data(fold, "D1")[0])
        n2 = len(s03.get_fold_data(fold, "D2")[0])
        assert n2 < n1


def test_fold_data_has_no_missing_features(s03):
    """모든 fold의 학습·검증 피처에 결측이 없다."""
    for cond in s03.DATA_CONDITIONS:
        for fold in s03.ACTIVE_FOLDS:
            Xtr, ytr, Xva, yva, _ = s03.get_fold_data(fold, cond)
            assert not Xtr.isna().any().any()
            assert not Xva.isna().any().any()
            assert len(Xtr) > 0 and len(Xva) > 0
            assert int(yva["y_cls"].sum()) > 0, "검증에 양성이 없으면 Task B 평가 불가"


def test_oof_positives_far_exceed_test(s03):
    """OOF 양성 합계가 테스트 28건보다 훨씬 많다 — 비교 주근거로 쓰는 이유."""
    total = int(s03.fold_matrix[s03.fold_matrix["조건"] == "D1"]["검증 양성"].sum())
    assert total >= 150
    assert total > 28 * 5


def test_features_arg_subsets_columns(s03):
    """`features` 인자로 부분집합을 넘기면 그대로 적용된다 (ablation 용)."""
    sub = s03.FEATURE_COLS[:5]
    Xtr, *_ = s03.get_fold_data(2, "D1", features=sub)
    assert list(Xtr.columns) == sub


@pytest.mark.parametrize("fid", ["F14", "F15"])
def test_chapter3_figures_exist(s03, project_root, fid):
    hits = list((project_root / "outputs" / "figures").glob(f"{fid}_*.png"))
    assert hits and hits[0].stat().st_size > 5000
