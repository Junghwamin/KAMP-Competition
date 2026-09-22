"""7~10장 테스트 — 오류분석 · 피크저감 · 창의성 · 패키징.

7장은 **입력이 OOF여야** 하고, 8장은 **입력이 전체기간 실측이어야** 한다.
이 두 규칙이 깨지면 보고서 3·4장의 근거가 무너지므로 최우선으로 검증한다.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import pytest


# ══════════════════════════════════════════════════════════════════════
# 7장 — 입력이 OOF인가
# ══════════════════════════════════════════════════════════════════════
def test_ch7_input_is_oof_not_test(s07):
    """7장 입력은 OOF다 — 테스트 구간(9월)이 섞이면 안 된다."""
    assert len(s07.OOF) > 1000
    assert s07.OOF.index.max() < pd.Timestamp("2021-09-01"), "테스트 구간이 섞였다"
    assert set(s07.OOF["fold"].unique()) == {2, 3, 4, 5}


def test_ch7_oof_positives_far_exceed_test(s07):
    """OOF 양성이 테스트 28건보다 훨씬 많다 — 조건별 분해가 가능한 이유."""
    assert int(s07.OOF["y_cls"].sum()) > 100


def test_permutation_importance_covers_all_features(s07):
    """permutation importance 가 전체 피처를 평가한다."""
    assert len(s07.perm_importance) == len(s07.FEATURE_COLS)
    assert s07.perm_importance["중요도(MAE 증가)"].notna().all()
    assert s07.perm_importance["중요도(MAE 증가)"].is_monotonic_decreasing


def test_permutation_has_korean_labels(s07):
    """보고서에 실을 한글 라벨이 붙는다."""
    assert s07.perm_importance["한글라벨"].notna().all()
    assert (s07.perm_importance["한글라벨"] != s07.perm_importance["변수"]).any()


def test_top_features_are_three_or_more(s07):
    """3장 도입부의 [상위변수 1~3] 을 채울 수 있다."""
    assert len(s07.TOP_FEATURES) >= 3
    assert len(set(s07.TOP_FEATURES)) == len(s07.TOP_FEATURES)


def test_importance_table_has_direction_and_meaning(s07):
    """3.1절 표에 방향(증가/감소)과 현장 의미가 채워진다."""
    t = s07.importance_tbl
    assert len(t) == 5
    assert t["현장 의미"].notna().all()
    assert (t["현장 의미"].str.len() > 5).all()


def test_interaction_table_has_four_rows_with_n(s07):
    """3.2절 상호작용 4종에 표본수가 병기된다."""
    t = s07.interaction_tbl
    assert len(t) == 4
    assert "n" in t.columns
    assert (t["n"] > 0).all()


def test_condition_table_has_seven_rows_with_n_and_ci(s07):
    """3.3절 표 7행에 n과 95% CI가 병기된다 (plan 필수 요구)."""
    t = s07.condition_tbl
    assert len(t) == 7
    assert {"n(조건 충족)", "n(그 외)", "95% CI(조건 충족)"} <= set(t.columns)
    assert (t["n(조건 충족)"] > 0).all()


def test_season_row_marked_as_limited(s07):
    """'계절' 행은 OOF가 7~8월뿐이라 비교 제한을 명시한다."""
    row = s07.condition_tbl[s07.condition_tbl["구분"] == "계절"].iloc[0]
    assert "제한" in str(row["비고"]) or "다름" in str(row["비고"])


def test_confusion_table_four_kinds(s07):
    """3.4절 TP/FN/FP/TN 4구분과 발생조건이 채워진다."""
    t = s07.confusion_tbl
    assert len(t) == 4
    assert set(t["구분"]) == {"정탐(TP)", "미탐지(FN)", "오경보(FP)", "정상판정(TN)"}
    assert t["건수"].sum() == len(s07.OOF)


def test_fn_fp_conditions_identified(s07):
    """FN·FP 집중조건이 문자열로 도출된다 (3.4절 [실제 조건] 채움)."""
    assert s07.FN_TOP_CONDITION != "-"
    assert s07.FP_TOP_CONDITION != "-"
    assert 0 <= s07.FN_TOP_SHARE_PCT <= 100


def test_failure_cases_are_distinct_days(s07):
    """3.5절 실패사례 3건이 서로 다른 날짜에서 선정된다."""
    t = s07.failure_tbl
    assert len(t) == 3
    days = [r.split(" ")[0] for r in t["시점 및 조건"]]
    assert len(set(days)) == 3
    assert (t["절대오차"] > 0).all()
    assert t["오류 원인"].notna().all()


def test_rules_are_depth3_and_data_derived(s07):
    """3.6절 규칙이 깊이 3 이하이고 실제 데이터에서 도출된다."""
    t = s07.rules_tbl
    assert 1 <= len(t) <= 3
    assert (t["피크 확률"] >= 0).all() and (t["피크 확률"] <= 1).all()
    assert (t["n"] >= 20).all(), "표본 20건 미만 규칙은 신뢰할 수 없다"
    # 깊이 3 이하 → 조건이 최대 3개
    for cond in t["피크 발생조건"]:
        assert cond.count("∧") <= 2


def test_rules_sorted_by_probability(s07):
    """규칙이 피크 확률 내림차순이다 (R1이 최고위험)."""
    assert s07.rules_tbl["피크 확률"].is_monotonic_decreasing


# ══════════════════════════════════════════════════════════════════════
# 8장 — 입력이 전체기간 실측인가
# ══════════════════════════════════════════════════════════════════════
def test_ch8_input_is_full_period_actual(s08):
    """8장은 전체기간 실측 peak15 위에서 계산한다 (예측값 아님)."""
    assert len(s08.PEAK15) == 6168
    assert s08.PEAK15.index.min() == pd.Timestamp("2021-01-01 00:00")
    assert s08.PEAK15.index.max() == pd.Timestamp("2021-09-14 23:00")
    assert float(s08.PEAK15.max()) == 222.0


def test_september_only_saving_is_zero(s08):
    """9월만으로는 기본요금 절감이 0원임을 수치로 증명한다 (정직성 근거)."""
    assert s08.SEPT_ONLY_SAVING == 0.0


def test_annual_billing_peak_is_222(s08):
    """청구 피크 = 3개월 롤링 최대의 최대 = 222 kW."""
    assert s08.BASE_BILLING_PEAK == 222.0


def test_monthly_max_matches_oracle(s08):
    """월별 최대 peak15 = 222/198/222/199/199/222/222/218/204."""
    mm = s08.tariff_tbl["월별최대"].tolist()
    assert mm == [222, 198, 222, 199, 199, 222, 222, 218, 204]


def test_response_function_is_pooled_and_significant(s08):
    """반응함수 β가 pooled 회귀에서 유의하게 추정된다."""
    assert s08.BETA > 0
    assert s08.BETA_DIAG["p-value"] < 0.05
    assert s08.BETA_DIAG["R²"] > 0.5
    assert s08.BETA_DIAG["n"] > 1000


def test_top_peaks_concentrate_in_specific_hours(s08):
    """상위 피크가 08·09·11시에 집중된다 — L1 설계의 근거."""
    hours = set(s08.TOP_PEAK_TIMES.hour)
    assert hours <= {8, 9, 10, 11, 12, 13}, f"예상 밖 시간대: {hours}"
    assert 8 in hours and 11 in hours


def test_l1a_alone_cannot_reduce_billing_peak(s08):
    """**핵심 발견**: 08·13시만 분산하면 Δ최대수요가 0이다.

    07-19 11시(222 kW)가 남기 때문이다. 청구 피크는 최대값이므로
    상위 사건을 하나라도 남기면 절감이 생기지 않는다.
    """
    l1a = s08.levers_tbl[s08.levers_tbl["레버"].str.startswith("L1a")].iloc[0]
    assert float(l1a["Δ최대수요(kW)"]) == 0.0
    assert float(l1a["기본요금 절감(원)"]) == 0.0


def test_covering_more_peak_hours_increases_saving(s08):
    """L1a < L1b < L1c 순으로 저감량이 커진다 (순차 포괄의 효과)."""
    t = s08.levers_tbl.set_index("레버")
    a = float(t.loc["L1a 기동 분산 (08·13시)", "Δ최대수요(kW)"])
    b = float(t.loc["L1b 기동 분산 (08·11·13시)", "Δ최대수요(kW)"])
    c = float(t.loc["L1c 기동 분산 (08·09·11·13시)", "Δ최대수요(kW)"])
    assert a < b < c


def test_production_shifting_has_weak_leverage(s08):
    """생산량 이동(L2)의 지렛대가 미미함을 수치로 확인한다."""
    l2 = s08.levers_tbl[s08.levers_tbl["레버"].str.startswith("L2")].iloc[0]
    l1c = s08.levers_tbl[s08.levers_tbl["레버"].str.startswith("L1c")].iloc[0]
    assert float(l2["Δ최대수요(kW)"]) < float(l1c["Δ최대수요(kW)"]) / 10


def test_all_levers_preserve_daily_production(s08):
    """일일 총 생산량이 보존된다 — 매출 손실이 없다는 주장의 근거.

    단 L4b(주말 이전)는 **다른 날로** 옮기므로 일 단위 보존이 깨진다.
    이는 설계상 의도된 것이며 표에 False 로 정직하게 표시된다.
    """
    t = s08.levers_tbl
    same_day = t[~t["레버"].str.startswith("L4b")]
    assert same_day["일생산량 보존"].all()
    l4b = t[t["레버"].str.startswith("L4b")].iloc[0]
    assert not bool(l4b["일생산량 보존"]), "주말 이전은 날짜를 넘으므로 False 여야 정직하다"


def test_l3_carryover_is_zero(s08):
    """L3 이월 잔량이 0이다 (plan 필수 검증 — 상한이 낮으면 생산 미달)."""
    assert abs(s08.L3_LEFTOVER) < 1e-6


def test_night_transfer_can_be_net_negative(s08):
    """L4 야간 이전은 인건비 때문에 순절감이 음수일 수 있다.

    '무조건 좋은 레버가 아님'을 수치로 보이는 것이 plan 의 요구다.
    """
    l4 = s08.levers_tbl[s08.levers_tbl["레버"].str.startswith("L4a")].iloc[0]
    assert float(l4["인건비 증가(원)"]) > 0
    assert float(l4["순절감액(원)"]) < float(l4["기본요금 절감(원)"])


def test_weekend_premium_is_zero_in_data(s08):
    """주말 할증이 이 데이터상 0임을 명시하고 파라미터로 분리한다."""
    assert s08.WEEKEND_LABOR_MULTIPLIER == 1.0
    assert s08.WEEKEND_PREMIUM_PARAM > 1.0


def test_scenarios_have_evidence_strength_labels(s08):
    """레버 표에 '모델기반/가정기반' 근거 강도가 구분된다."""
    assert "근거 강도" in s08.levers_tbl.columns
    kinds = " ".join(s08.levers_tbl["근거 강도"])
    assert "가정기반" in kinds and "모델기반" in kinds


def test_sensitivity_covers_rate_and_alpha(s08):
    """기본요금 단가 ±30% × α 3수준 = 9조합을 평가한다."""
    t = s08.sensitivity_tbl
    assert len(t) == 9
    assert set(t["기본요금단가 배수"]) == {0.7, 1.0, 1.3}
    assert set(t["α"]) == {0.10, 0.20, 0.30}


def test_lever_priority_is_stable_under_sensitivity(s08):
    """가정을 ±30% 흔들어도 최적 조합이 뒤집히지 않는다."""
    assert s08.PRIORITY_STABLE, (
        f"최적 조합이 바뀐다: {s08.sensitivity_tbl['최적 조합'].unique().tolist()}"
    )


def test_protocol_links_prediction_to_action(s08):
    """운영 프로토콜이 예측 시점부터 현장 조치까지 이어진다."""
    t = s08.protocol_tbl
    assert len(t) >= 6
    assert set(t["주체"]) <= {"자동", "현장", "분석"}
    assert any("24:00" in x for x in t["시점"]), "예측 원점이 프로토콜에 없다"


def test_ch4_draft_generated_with_key_content(s08, project_root):
    """4장 본문 초안이 핵심 내용을 담고 생성된다."""
    p = project_root / "outputs" / "report_ch4_draft.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert len(text) > 2000
    for kw in ("L1", "L2", "L3", "L4", "순절감액", "민감도", "운영 프로토콜"):
        assert kw in text, kw
    # 9월 단독 무효 사실을 반드시 명시
    assert "9월" in text and "영향" in text


# ══════════════════════════════════════════════════════════════════════
# 9장 — 창의성
# ══════════════════════════════════════════════════════════════════════
def test_creativity_has_five_items_linked_to_chapters(s09):
    """차별점 5종이 각각 2·3장 절에 연결된다."""
    t = s09.creativity_tbl
    assert len(t) == 5
    assert t["대응 절"].notna().all()
    assert t["측정된 근거"].str.len().min() > 20


def test_ch5_draft_includes_honest_limits(s09, project_root):
    """5장 초안이 한계를 수치와 함께 명시한다 (정직성 가점)."""
    p = project_root / "outputs" / "report_ch5_draft.md"
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert "한계" in text
    for kw in ("유의", "Clopper-Pearson", "9월"):
        assert kw in text, kw


# ══════════════════════════════════════════════════════════════════════
# 10장 — 패키징
# ══════════════════════════════════════════════════════════════════════
def test_prediction_file_schema(s10, project_root):
    """제출 규격: 336행 · 결측 0 · 9컬럼 · utf-8-sig."""
    p = project_root / "outputs" / "predictions_test_336h.csv"
    assert p.exists()
    d = pd.read_csv(p, encoding="utf-8-sig")
    assert len(d) == 336
    assert int(d.isna().sum().sum()) == 0
    assert list(d.columns) == [
        "datetime", "y_avg_true", "y_avg_pred", "y_peak_true", "y_peak_pred",
        "peak_prob", "peak_pred_label", "peak_true_label", "model_name",
    ]
    assert set(d["peak_pred_label"].unique()) <= {0, 1}
    assert int(d["peak_true_label"].sum()) == 28


def test_prediction_file_has_bom(s10, project_root):
    """Excel 한글 호환을 위해 BOM 포함으로 저장된다."""
    raw = (project_root / "outputs" / "predictions_test_336h.csv").read_bytes()[:3]
    assert raw == b"\xef\xbb\xbf"


def test_tbd_table_covers_all_placeholders(s10):
    """채움표가 보고서 플레이스홀더를 빠짐없이 담는다."""
    t = s10.tbd_tbl
    assert len(t) >= 35
    assert t["채움값"].notna().all()
    assert (t["채움값"].astype(str).str.len() > 0).all()
    # 어떤 채움값에도 TBD 가 남아 있으면 안 된다
    assert not t["채움값"].astype(str).str.contains("TBD", case=False).any()


def test_placeholder_scanner_catches_all_variants(s10):
    """정규식이 `[TBD]` 변형을 모두 잡는다 (단순 grep 이 놓치는 것들)."""
    sample = "a [TBD] b [TBDd] c [TBDDd] d [Tbdd] e [TBdd] f TBd g [최종모델명 ] h"
    hits = s10.scan_placeholders(sample)
    joined = " ".join(hits)
    for variant in ("[TBD]", "[TBDd]", "[TBDDd]", "[Tbdd]", "[TBdd]", "[최종모델명 ]"):
        assert variant in joined, f"{variant} 를 놓쳤다"
    assert any("TBd" in h for h in hits), "대괄호 없는 TBd 를 놓쳤다"


def test_chapter_checklist_all_pass(s10):
    """장별 충족 체크리스트가 전부 통과한다 (4·5·6장 30점 포함)."""
    assert s10.chapter_checks["충족"].all(), (
        s10.chapter_checks[~s10.chapter_checks["충족"]].to_dict("records")
    )
    summary = s10.chapter_checks.attrs["summary"]
    assert summary.all()
    assert len(summary) == 6


def test_chapter_checklist_covers_empty_chapters(s10):
    """`[TBD]` 토큰이 0개인 4·5·6장이 체크리스트에 포함된다.

    정규식 점검만으로는 30점이 통째로 비어도 통과하므로 이 검사가 필수다.
    """
    chapters = set(s10.chapter_checks["장"])
    assert any("4장" in c for c in chapters)
    assert any("5장" in c for c in chapters)
    assert any("6장" in c for c in chapters)


def test_final_gate_all_pass(s10):
    """최종 게이트 전 항목 통과."""
    assert s10.gate_final["통과"].all(), (
        s10.gate_final[~s10.gate_final["통과"]]["점검"].tolist()
    )


def test_no_privacy_leak_in_submission_files(s10):
    """제출 대상 파일에 개인정보·절대경로가 없다 (블라인드 평가)."""
    assert len(s10.privacy_blocking) == 0, s10.privacy_blocking.to_dict("records")


def test_no_forbidden_patterns(s10):
    """이중 축(twinx)과 `freq='H'` 실제 호출이 0건이다."""
    assert len(s10.forbidden_scan) == 0, s10.forbidden_scan.to_dict("records")


def test_requirements_pinned_exactly(s10, project_root):
    """requirements.txt 가 `==` 로 완전 핀되고 미사용 패키지를 제외한다."""
    text = (project_root / "requirements.txt").read_text(encoding="utf-8")
    pinned = [l for l in text.splitlines() if l and not l.startswith("#")]
    assert any("pandas==" in l for l in pinned)
    assert any("lightgbm==" in l for l in pinned)
    # 실사용 없는 패키지는 제외
    assert "torch" not in text and "ortools" not in text
    assert "python-pptx" in text


def test_readme_covers_chapter6_requirements(s10, project_root):
    """README(=6장 본문)가 실행절차·환경·구성·재현성을 모두 담는다."""
    text = (project_root / "README.md").read_text(encoding="utf-8")
    for kw in ("실행 방법", "파일 구성", "재현성", "게이트", "requirements.txt"):
        assert kw in text, kw


def test_all_39_figures_with_index(s10, project_root):
    """그림 39장이 모두 생성되고 인덱스와 개수가 일치한다."""
    figdir = project_root / "outputs" / "figures"
    pngs = sorted(figdir.glob("F*.png"))
    assert len(pngs) == 39, f"{len(pngs)}장"
    idx = pd.read_csv(figdir / "figure_index.csv", encoding="utf-8-sig")
    assert len(idx) == 39
    assert idx["fid"].is_unique
    for i in range(1, 40):
        assert any(p.name.startswith(f"F{i:02d}_") for p in pngs), f"F{i:02d} 누락"


def test_source_tables_accompany_figures(s10, project_root):
    """🔵 본문 그림에 소스 표가 동반된다 (대비 WARN 완화 조건)."""
    tbl = project_root / "outputs" / "tables"
    idx = pd.read_csv(
        project_root / "outputs" / "figures" / "figure_index.csv", encoding="utf-8-sig"
    )
    with_src = idx[idx["소스표"].notna() & (idx["소스표"].astype(str) != "")]
    assert len(with_src) >= 30, f"소스표 동반 그림 {len(with_src)}장"
    for name in with_src["소스표"]:
        assert (tbl / name).exists(), name


def test_pptx_generated(s10, project_root):
    """발표자료 골격이 생성된다."""
    p = project_root / "outputs" / "발표자료.pptx"
    assert p.exists() and p.stat().st_size > 20000
