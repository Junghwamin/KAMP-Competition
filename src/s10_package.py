# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    FAST,
    FIG_DIR,
    FIGURE_INDEX_PATH,
    FONT_NAME,
    HAS_TF,
    OUTPUT_DIR,
    SEED,
    TBL_DIR,
    display,
    env_versions,
    save_table,
)
from s01_diagnose import (  # noqa: F401
    N_SURPLUS_DAYS,
    N_TOTAL_DAYS,
    N_UNIQUE_PROFILES,
    THETA,
    TEST_END,
    TEST_START,
    df,
    operating_calendar,
)
from s02_features import FEATURE_COLS, N_WARMUP  # noqa: F401
from s03_split import ACTIVE_FOLDS, CALENDAR_RULE_FP, calendar_rule_metrics  # noqa: F401
from s05_models import HPO_N_TRIALS, cv_results, test_results  # noqa: F401
from s06_eval import (  # noqa: F401
    BEST_BASELINE_MAE,
    BEST_BASELINE_NAME,
    CLF_FN,
    CLF_FP,
    CLF_RECALL,
    CLF_TP,
    FINAL_F1,
    FINAL_MAE,
    FINAL_MODEL_NAME,
    FINAL_PEAK_MAE,
    FINAL_RECALL,
    IMPROVE_MAE_PCT,
    IMPROVE_PEAKMAE_PCT,
    INFER_SEC,
    MAE_IMPROVEMENT_SIGNIFICANT,
    SURROGATE_FIDELITY,
    TEST_POSITIVES,
    interpretation_tbl,
    peak_tbl,
    rank_preservation,
)
from s07_analysis import (  # noqa: F401
    BEST_CONDITION_NAME,
    FN_TOP_CONDITION,
    FN_TOP_SHARE_PCT,
    FP_TOP_CONDITION,
    TOP_FEATURES,
    TOP_INTERACTION,
    WORST_CONDITIONS,
    WORST_CONDITION_NAME,
    WORST_COND_EXCESS_PCT,
    OOF,
)
from s08_simulation import (  # noqa: F401
    BETA,
    PEAK_REDUCTION_KW,
    SEPT_ONLY_SAVING,
    scenarios_tbl,
)
from s09_creativity import creativity_tbl  # noqa: F401
import re
import numpy as np
import pandas as pd

# %% [markdown]
# ## 10. 산출물·재현성·제출 패키징
#
# 보고서 **6장 코드 구성 및 재현성(10점)** 의 근거를 생성하고, 제출물을 만든다.

# %% [markdown]
# ### 10.1 테스트 예측결과 파일
#
# - **목적**: 제출물 "테스트데이터 예측결과 파일"을 규격에 맞게 생성한다.
# - **보고서 대응절**: 6장
# - **산출물**: `outputs/predictions_test_336h.csv`
#
# 규격: **336행 · 결측 0 · `utf-8-sig`**, 컬럼
# `datetime, y_avg_true, y_avg_pred, y_peak_true, y_peak_pred, peak_prob, peak_pred_label, peak_true_label, model_name`

# %%
PREDICTION_PATH = OUTPUT_DIR / "predictions_test_336h.csv"


def build_prediction_file() -> pd.DataFrame:
    """제출용 예측결과 파일을 만든다.

    피크 확률은 '피크 직접분류' 모델에서, 판정 라벨은 최종모델의 τ 적용 결과에서 가져온다.
    테스트 336행은 어떤 마스크와도 무관하게 **전부 유지**한다.
    """
    te = test_results[FINAL_MODEL_NAME]
    tau = cv_results[FINAL_MODEL_NAME]["tau"]
    clf = test_results.get("피크 직접분류")
    prob = clf["prob"] if clf is not None and clf["prob"] is not None else np.full(len(te["index"]), np.nan)

    out = pd.DataFrame(
        {
            "datetime": pd.DatetimeIndex(te["index"]).strftime("%Y-%m-%d %H:%M:%S"),
            "y_avg_true": np.round(te["y_avg"], 4),
            "y_avg_pred": np.round(te["pred_avg"], 4),
            "y_peak_true": np.round(te["y_peak"], 4),
            "y_peak_pred": np.round(te["pred_peak"], 4),
            "peak_prob": np.round(prob, 6),
            "peak_pred_label": (te["pred_peak"] >= tau).astype(int),
            "peak_true_label": te["y_cls"].astype(int),
            "model_name": FINAL_MODEL_NAME,
        }
    )
    return out


predictions = build_prediction_file()
predictions.to_csv(PREDICTION_PATH, index=False, encoding="utf-8-sig")
print(f"── 예측결과 파일 생성: {PREDICTION_PATH.name} ──")
print(f"  행 수 {len(predictions)} / 결측 {int(predictions.isna().sum().sum())}건 / 인코딩 utf-8-sig")
display(predictions.head(3))

# %% [markdown]
# ### 10.2 보고서 채움표 · 문장 치환사전 · 장별 PASS/FAIL
#
# - **목적**: 보고서의 모든 `[TBD]` 를 실측값으로 채우고, **장별 충족 여부**를 판정한다.
# - **보고서 대응절**: 전 장
# - **산출물**: `outputs/report_tbd_filled.md`, `ch6_tbd_table.csv`
#
# > ### ⚠️ 빈칸 점검은 **2단계**여야 한다
# >
# > **① 정규식 점검만으로는 부족하다.** 단순 `[TBD]` 리터럴 grep 은
# > `[TBDd]`·`[TBDDd]`·`[Tbdd]`·`[TBdd]`·대괄호 없는 `TBd`·내부 공백이 있는 `[최종모델명 ]` 을
# > 전부 놓친다. 정규식은 `\[?T[Bb][Dd]+\]?` 와 `\[[^\]]*\]` 두 개를 함께 쓴다.
# >
# > **② 4·5·6장은 `[TBD]` 토큰이 0개다.** 따라서 정규식만 돌리면
# > **30점이 통째로 비어 있어도 통과한다.** 장별 충족 체크리스트가 반드시 필요하다.

# %%
def build_tbd_table() -> pd.DataFrame:
    """보고서 플레이스홀더 → 실측값 채움표."""
    rows = [
        # 1쪽 내용요약
        ("1쪽 내용요약", "[TBD]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1 유의성 검정"),
        ("1쪽 내용요약", "[TBD] (피크 Recall)", f"{FINAL_RECALL:.3f}", "6.2 피크탐지표"),
        ("1쪽 내용요약", "[TBD] (피크 F1)", f"{FINAL_F1:.3f}", "6.2 피크탐지표"),
        ("1쪽 내용요약", "[TBD] kW (최대수요 저감)", f"{PEAK_REDUCTION_KW:.1f} kW", "8.3 시나리오 (전체기간 기준)"),
        # 2장 도입
        ("2장 도입", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4 스코어카드"),
        ("2장 도입", "[TBDd]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2장 도입", "[TBDDd] (Recall)", f"{FINAL_RECALL:.3f}", "6.2"),
        # 2.5절
        ("2.5절", "[TBDd]회 탐색", f"{HPO_N_TRIALS}회", "5.1·5.4 Optuna 이력"),
        # 2.6절
        ("2.6절 표", "[최종모델]", FINAL_MODEL_NAME, "6.4"),
        ("2.6절 본문", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4"),
        ("2.6절 본문", "[TBDD] (최종 MAE)", f"{FINAL_MAE:.3f}", "6.1"),
        ("2.6절 본문", "[TBDd] (최선 베이스라인 MAE)", f"{BEST_BASELINE_MAE:.3f} ({BEST_BASELINE_NAME})", "6.1"),
        ("2.6절 본문", "[Tbdd]% (개선율)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2.6절 본문", "[TBDd]% (Peak-MAE 감소)", f"{IMPROVE_PEAKMAE_PCT:.2f}%", "6.1"),
        # 2.7절
        ("2.7절 본문", "[TBDd] (직접분류 Recall)", f"{CLF_RECALL:.3f}", "6.2"),
        ("2.7절 본문", "[TBD] (전체 피크 건수)", f"{TEST_POSITIVES}건", "1.6 θ 적용"),
        ("2.7절 본문", "[TBDd] (탐지 건수)", f"{CLF_TP}건", "6.2"),
        ("2.7절 본문", "[TBDd] (오경보)", f"{CLF_FP}건", "6.2"),
        ("2.7절 본문", "[TBdd] (미탐지)", f"{CLF_FN}건", "6.2"),
        # 2.9절
        ("2.9절", "[최종모델명 ]", FINAL_MODEL_NAME, "6.4"),
        ("2.9절", "[TBD]% (MAE 개선)", f"{IMPROVE_MAE_PCT:.2f}%", "6.1"),
        ("2.9절", "[TBDd] (Recall)", f"{FINAL_RECALL:.3f}", "6.2"),
        ("2.9절", "[TBDd]초 (추론시간)", f"{INFER_SEC:.2f}초", "6.4 (336h 배치, 10회 중앙값)"),
        # 3장 도입
        ("3장 도입", "[상위변수 1]", TOP_FEATURES[0], "7.1 permutation"),
        ("3장 도입", "[상위변수 2]", TOP_FEATURES[1], "7.1 permutation"),
        ("3장 도입", "[상위변수 3]", TOP_FEATURES[2], "7.1 permutation"),
        ("3장 도입", "[가동전환, 휴일 다음날, …]", ", ".join(WORST_CONDITIONS), "7.3 조건별 오차"),
        ("3장 도입", "[TBDd]% (초과율)", f"{WORST_COND_EXCESS_PCT:.1f}%", "7.3"),
        # 3.1·3.2절
        ("3.1절", "[예:24시간 전 전력]", TOP_FEATURES[0], "7.1"),
        ("3.1절", "[예: 생산량]", TOP_FEATURES[1], "7.1"),
        ("3.2절", "[실제 확인된 상호작용]", TOP_INTERACTION, "7.2"),
        # 3.3·3.4절
        ("3.3절", "[실제 조건] (최대오차)", WORST_CONDITION_NAME, "7.3"),
        ("3.3절", "[TBD]% (평균 대비)", f"{WORST_COND_EXCESS_PCT:.1f}%", "7.3"),
        ("3.3절", "[실제 조건] (안정)", BEST_CONDITION_NAME, "7.3"),
        ("3.4절", "[실제 조건] (FN 집중)", FN_TOP_CONDITION, "7.4"),
        ("3.4절", "[TBDd]% (FN 비중)", f"{FN_TOP_SHARE_PCT:.1f}%", "7.4"),
        ("3.4절", "[실제조건] (FP 집중)", FP_TOP_CONDITION, "7.4"),
    ]
    return pd.DataFrame(rows, columns=["위치", "플레이스홀더", "채움값", "산출 근거(노트북 절)"])


tbd_tbl = build_tbd_table()
save_table(tbd_tbl, "ch6_tbd_table")
print(f"── 보고서 채움표 {len(tbd_tbl)}항목 ──")
display(tbd_tbl)


def scan_placeholders(text: str) -> list[str]:
    """2단계 빈칸 점검의 ① — 정규식으로 잔여 플레이스홀더를 찾는다.

    단순 `[TBD]` grep 이 놓치는 변형을 모두 잡는다.
    """
    WHITELIST = {"[O]", "[X]", "[발견 1]", "[발견 2]", "[발견 3]"}
    hits = []
    for pat in (r"\[?T[Bb][Dd]+\]?", r"\[[^\]\n]{0,40}\]"):
        for m in re.finditer(pat, text):
            tok = m.group()
            if tok in WHITELIST:
                continue
            # 마크다운 링크·각주는 제외
            if re.fullmatch(r"\[\d+\]", tok):
                continue
            hits.append(tok)
    return sorted(set(hits))


CHAPTER_REQUIREMENTS = {
    "1장 데이터 이해 및 진단": [
        ("변수사전 18컬럼", lambda: (TBL_DIR / "ch1_variable_dict.csv").exists()),
        ("품질 6기준 진단표", lambda: (TBL_DIR / "ch1_quality.csv").exists()),
        ("프로파일 복제 진단", lambda: N_SURPLUS_DAYS == 115),
        ("가동 캘린더", lambda: (TBL_DIR / "ch1_calendar.csv").exists()),
        ("θ 확정", lambda: THETA == 187.0),
    ],
    "2장 AI 예측모델 개발 및 성능평가": [
        ("2.2 베이스라인 결함표", lambda: (TBL_DIR / "ch4_defects.csv").exists()),
        ("2.6 회귀 성능표", lambda: (TBL_DIR / "ch2_regression.csv").exists()),
        ("2.7 피크탐지표 + τ + 달력규칙 행", lambda: "τ" in pd.read_csv(TBL_DIR / "ch2_peak_detection.csv", encoding="utf-8-sig").columns),
        ("2.8 Ablation 5행", lambda: len(pd.read_csv(TBL_DIR / "ch2_ablation.csv", encoding="utf-8-sig")) == 5),
        ("2.9 스코어카드", lambda: (TBL_DIR / "ch2_scorecard.csv").exists()),
        ("D1/D2 대비표", lambda: (TBL_DIR / "ch2_d1_vs_d2.csv").exists()),
    ],
    "3장 영향요인 및 오류분석": [
        ("3.1 상위 5변수", lambda: len(pd.read_csv(TBL_DIR / "ch3_importance.csv", encoding="utf-8-sig")) == 5),
        ("3.2 상호작용 4종", lambda: len(pd.read_csv(TBL_DIR / "ch3_interaction.csv", encoding="utf-8-sig")) == 4),
        ("3.3 조건별 오차 7행", lambda: len(pd.read_csv(TBL_DIR / "ch3_condition_mae.csv", encoding="utf-8-sig")) == 7),
        ("3.4 혼동행렬 4구분", lambda: len(pd.read_csv(TBL_DIR / "ch3_confusion.csv", encoding="utf-8-sig")) == 4),
        ("3.5 실패사례 3건", lambda: len(pd.read_csv(TBL_DIR / "ch3_failures.csv", encoding="utf-8-sig")) == 3),
        ("3.6 규칙 R1~R3", lambda: len(pd.read_csv(TBL_DIR / "ch3_rules.csv", encoding="utf-8-sig")) >= 1),
    ],
    "4장 현장 활용방안 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch4_draft.md").exists()),
        ("레버 L1~L4 전부", lambda: len(pd.read_csv(TBL_DIR / "ch4_levers.csv", encoding="utf-8-sig")) >= 4),
        ("순절감액 산출", lambda: "순절감액(원)" in pd.read_csv(TBL_DIR / "ch4_scenarios.csv", encoding="utf-8-sig").columns),
        ("민감도 ±30%", lambda: (TBL_DIR / "ch4_sensitivity.csv").exists()),
        ("운영 프로토콜", lambda: (TBL_DIR / "ch4_protocol.csv").exists()),
    ],
    "5장 창의성 및 차별성 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch5_draft.md").exists()),
        ("차별점 5종", lambda: len(creativity_tbl) == 5),
        ("2·3장 결과 연결", lambda: creativity_tbl["대응 절"].notna().all()),
    ],
    "6장 코드 구성 및 재현성 (10점)": [
        ("본문 초안 생성", lambda: (OUTPUT_DIR / "report_ch6_draft.md").exists()),
        ("예측결과 파일 336행", lambda: len(predictions) == 336),
        ("requirements.txt", lambda: (OUTPUT_DIR.parent / "requirements.txt").exists()),
        ("환경 증거(requirements_generated)", lambda: (OUTPUT_DIR / "requirements_generated.txt").exists()),
        ("그림 인덱스", lambda: FIGURE_INDEX_PATH.exists()),
        ("채움표", lambda: len(tbd_tbl) >= 30),
    ],
}


def check_chapters() -> pd.DataFrame:
    """2단계 빈칸 점검의 ② — 장별 충족 체크리스트."""
    rows = []
    for chapter, checks in CHAPTER_REQUIREMENTS.items():
        for label, fn in checks:
            try:
                ok = bool(fn())
            except Exception as e:
                ok = False
                label = f"{label} (오류: {type(e).__name__})"
            rows.append({"장": chapter, "요건": label, "충족": ok})
    out = pd.DataFrame(rows)
    summary = out.groupby("장")["충족"].all().rename("PASS")
    out.attrs["summary"] = summary
    return out


chapter_checks = check_chapters()
save_table(chapter_checks, "ch6_chapter_checklist")
print("\n── 장별 충족 체크리스트 ──")
display(chapter_checks.attrs["summary"].to_frame())
_failed = chapter_checks[~chapter_checks["충족"]]
if len(_failed):
    print("\n  ⚠️ 미충족 요건:")
    display(_failed)
else:
    print("\n  ✅ 전 장 요건 충족")

# %% [markdown]
# ### 10.3 `report_tbd_filled.md` 생성 — 채움표 + 치환사전 + plan 대비 정정

# %%
# plan.md 예상치와 실측이 다른 항목 — 보고서가 plan 수치로 쓰여 있으면 어긋난다
PLAN_DELTAS = [
    ("fold3 휴무 비중", "29%", f"{float(pd.read_csv(TBL_DIR / 'ch3_folds.csv', encoding='utf-8-sig').set_index('fold').loc[3, '휴무비중'].strip('%')):.0f}%",
     "가동 캘린더를 `일생산량==0` 으로 판정하면 일요일·증강불일치일이 휴무로 더 많이 잡힌다"),
    ("fold4 휴무 비중", "36%", f"{float(pd.read_csv(TBL_DIR / 'ch3_folds.csv', encoding='utf-8-sig').set_index('fold').loc[4, '휴무비중'].strip('%')):.0f}%", "동일"),
    ("OOF 양성 합계", "185건", f"{int(OOF['y_cls'].sum())}건",
     "계측정지 17행·ERP결측 48행을 학습·평가에서 제외하고 fold1을 폐기한 결과"),
    ("토요일 평균전력", "51.1", "51.3", "08-28~29 계측정지 17행을 시간보간한 결과 (+0.2)"),
    ("일요일 평균전력", "41.3", "41.6", "동일 (+0.3)"),
    ("08시 피크율", "24.7%", "24.5%", "동일"),
    ("2/11·3/1 판정", "정상 가동일(공휴일 리스트 오류)", "공휴일 맞음 — 전력만 증강 덮어쓰기",
     "두 날 모두 생산량·공장인원이 0이고 프로파일 복제그룹에 속한다 → 전력 단일 채널 판정은 위험"),
]


def write_tbd_filled() -> str:
    """채움표·치환사전·정정사항·장별 PASS 를 하나의 문서로 만든다."""
    md = f"""# 보고서 빈칸 채움표 및 문장 치환사전

> 생성 시점의 노트북 실행 결과다. 노트북을 다시 실행하면 이 문서도 갱신된다.
> **FAST 모드 실행 여부: {FAST}** (True 면 탐색·에폭이 축소된 값이므로 제출 전 FULL 재실행 필요)

## 1. 최종 결과 요약

| 항목 | 값 |
|---|---|
| 최종모델 | **{FINAL_MODEL_NAME}** |
| 최선 베이스라인 | {BEST_BASELINE_NAME} (MAE {BEST_BASELINE_MAE:.3f}) |
| 테스트 MAE | {FINAL_MAE:.3f} ({IMPROVE_MAE_PCT:+.2f}%) |
| 테스트 Peak-MAE | {FINAL_PEAK_MAE:.3f} ({IMPROVE_PEAKMAE_PCT:+.2f}%) |
| 테스트 Recall / F1 | {FINAL_RECALL:.3f} / {FINAL_F1:.3f} |
| MAE 개선 유의성 | {'유의' if MAE_IMPROVEMENT_SIGNIFICANT else '**유의하지 않음** (CI가 0 포함)'} |
| 336시간 배치 추론 | {INFER_SEC:.2f}초 (10회 중앙값) |
| 최대수요 저감 (전체기간) | {PEAK_REDUCTION_KW:.1f} kW |
| HPO 탐색 횟수 | {HPO_N_TRIALS}회 |
| θ (피크 정의) | {THETA:g} (고정) |

## 2. 플레이스홀더 치환표 ({len(tbd_tbl)}항목)

| 위치 | 플레이스홀더 | → 채움값 | 산출 근거 |
|---|---|---|---|
"""
    for _, r in tbd_tbl.iterrows():
        md += f"| {r['위치']} | `{r['플레이스홀더']}` | **{r['채움값']}** | {r['산출 근거(노트북 절)']} |\n"

    md += """
## 3. 본문 문장 교정 (실측과 어긋나는 서술)

보고서 초안에 이미 쓰여 있으나 **실측과 다른** 서술이다. 반드시 고쳐야 한다.

"""
    fixes = [
        ("1.1절", "피크 기준 서술",
         f"`peak15 = max(15분, 30분, 45분, 60분)` 정의를 각주로 명시한다. "
         f"`15분` 단일 컬럼으로는 q95/q99/max 가 173.4/188/207 로 보고서의 187/201/222 와 불일치한다."),
        ("1.3절", "야간 피크 서술",
         "'야간·심야 피크 0건' → **'정상 가동일 기준 야간 피크 0건 "
         "(야간 피크 7건은 전부 ERP 결측일 7/13·7/15에 귀속)'**. "
         "이 문장이 3.6절 규칙과 4장 L4 레버의 근거이므로 그대로 두면 안 된다."),
        ("1.4절", "유일성 행",
         f"'날짜와 시간이 중복된 5건' → **'(날짜,시간) 중복 5건 + 24시간 전력 프로파일 복제 "
         f"{N_SURPLUS_DAYS}일({N_SURPLUS_DAYS / N_TOTAL_DAYS:.1%})'**"),
        ("1.4절", "무결성 지수 서술",
         "'무결성 지수는 33.3%에서 100%로 개선' → **'복제 구조를 진단하고 학습 설계에 반영'**. "
         "복제는 오류가 아니라 증강 구조이므로 '개선'으로 표현하면 부정확하다."),
        ("1.4절", "결측 서술 보강",
         "**'결측이 센서측(08-28 18시~08-29 10시, 17시간)과 ERP측(7/13·7/15, 48시간)의 "
         "양방향으로 존재함을 확인'** 을 추가한다. 양방향 결측을 대칭적으로 진단한 것이 "
         "서면평가 1번(검증전략)의 강한 근거다."),
        ("1.2절", "변수 단위 정정",
         "`공장인원`은 인원수가 아니라 **0~48.39 연속 실수(정규화 투입 공수)**, "
         "`인건비`는 금액이 아니라 **교대 배수(9~17시 1.0, 그 외 1.5)** 임을 명시한다."),
        ("1.5절", "기상변수 가정",
         "기상 실측값을 **D+1 예보의 대리(proxy)** 로 사용했음을 명시하고, "
         "기상 ablation 결과를 '예보오차에 대한 성능 하한'으로 해석한다."),
        ("1.6절", "학습 표본 한계",
         f"복제로 인해 **실질 학습 표본이 명목의 약 {N_UNIQUE_PROFILES / N_TOTAL_DAYS:.0%}** 임을 명시하고, "
         f"워밍업 탈락 {N_WARMUP}행을 반영한 실제 학습 구간을 기재한다."),
        ("2.1절", "교차검증 fold 수",
         f"'Rolling-origin 5-fold' → **'Rolling-origin {len(ACTIVE_FOLDS)}-fold "
         "(fold1은 검증일 50%가 학습구간과 동일 전력 프로파일이라 폐기)'**"),
        ("2.1절", "Task B 주지표",
         f"'Task B 주지표 Recall' → **'Task B 주지표 F1 / PR-AUC'**. "
         f"달력규칙 한 줄이 Recall 1.000(FP {CALENDAR_RULE_FP})을 내므로 Recall 로는 "
         "어떤 모델도 이길 수 없다. 주지표를 'Recall 유지 + 오경보 최소화'로 재정의한다."),
        ("4장", "절감액 해석 한계",
         f"**'테스트 구간(9월) 단독으로는 연간 기본요금에 영향이 없다'** 를 명시한다. "
         f"9월 peak15 를 0으로 낮추는 극단 가정에서도 절감액은 {SEPT_ONLY_SAVING:,.0f}원이다."),
    ]
    md += "| 절 | 대상 | 교정 내용 |\n|---|---|---|\n"
    for sec, target, fix in fixes:
        md += f"| {sec} | {target} | {fix} |\n"

    md += "\n### 3.1 6장에서 도출된 추가 교정\n\n"
    for _, r in interpretation_tbl.iterrows():
        md += f"- **{r['항목']}** (실측: {r['실측']})\n  → {r['보고서 서술 교정']}\n\n"

    md += """## 4. plan.md 예상치 대비 실측 정정

`plan.md` 에 적힌 예상 수치와 실제 실행 결과가 다른 항목이다.
보고서가 plan 수치로 쓰여 있으면 노트북 출력과 어긋나므로 **아래 실측값으로 맞춘다.**

| 항목 | plan.md | 실측 | 원인 |
|---|---|---|---|
"""
    for item, planv, actual, cause in PLAN_DELTAS:
        md += f"| {item} | {planv} | **{actual}** | {cause} |\n"

    md += "\n## 5. 장별 충족 판정 (PASS/FAIL)\n\n"
    md += "> ⚠️ 4·5·6장은 `[TBD]` 토큰이 0개라 정규식 점검만으로는 통째로 비어도 통과한다.\n"
    md += "> 그래서 아래 요건 기반 판정이 필수다.\n\n"
    md += "| 장 | 판정 |\n|---|---|\n"
    for ch, ok in chapter_checks.attrs["summary"].items():
        md += f"| {ch} | {'✅ PASS' if ok else '❌ FAIL'} |\n"
    md += "\n<details><summary>요건별 상세</summary>\n\n| 장 | 요건 | 충족 |\n|---|---|---|\n"
    for _, r in chapter_checks.iterrows():
        md += f"| {r['장']} | {r['요건']} | {'✅' if r['충족'] else '❌'} |\n"
    md += "\n</details>\n\n"

    md += """## 6. 사람이 수행할 항목 (노트북 밖)

1. 설문 응답 후 완료화면 캡처 → 보고서 10쪽 삽입
2. 발표자료 PDF·PPT 최종화 (`outputs/발표자료.pptx` 골격 활용)
3. 보고서 "작성 요령 ※ 작성 후 삭제" 블록 **전부 삭제**
4. 1쪽 서명·날짜 기입
5. 블라인드 규정 최종 확인 (소속·학교·로고 없음, 성명·팀명만)
6. 그림 39장 렌더 육안 점검 (라벨 충돌·축 잘림·한글 깨짐·범례 겹침)
7. 보고서 PDF 내보내기 후 `/Author` 메타데이터 확인·제거 (HWP 내보내기 시 계정명이 박힌다)
"""
    return md


_tbd_md = write_tbd_filled()
(OUTPUT_DIR / "report_tbd_filled.md").write_text(_tbd_md, encoding="utf-8")
_residual = scan_placeholders(_tbd_md.split("## 2. 플레이스홀더 치환표")[0])
print(f"── report_tbd_filled.md 생성 ({len(_tbd_md):,}자) ──")
print(f"  요약부 잔여 플레이스홀더: {_residual if _residual else '0건'}")

# %% [markdown]
# ### 10.4 requirements.txt · README(= 6장 본문)
#
# - **목적**: 심사위원이 전처리부터 결과생성까지 실행할 수 있는 문서를 만든다.
# - **보고서 대응절**: 6장 전체
# - **산출물**: `requirements.txt`, `README.md`, `outputs/report_ch6_draft.md`

# %%
PROJECT_ROOT = OUTPUT_DIR.parent


def write_requirements() -> str:
    """버전을 `==` 로 완전 핀한 requirements.txt.

    `torch`·`ortools` 는 실사용이 없으므로 제외한다.
    공휴일은 외부 패키지 대신 2021 상수 리스트를 쓴다(폐쇄망 심사환경 대응).
    """
    v = env_versions().set_index("항목")["버전"].to_dict()
    lines = ["# 제6회 K-인공지능 제조데이터 분석 경진대회 ⑤ 자원 최적화",
             f"# Python {v.get('python', '3.11')}", ""]
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib",
                "lightgbm", "optuna", "shap", "statsmodels"):
        ver = v.get(pkg, "")
        if ver and ver != "미설치":
            lines.append(f"{pkg}=={ver}")
    if v.get("tensorflow", "미설치") != "미설치" and not v["tensorflow"].startswith("미설치"):
        lines.append(f"tensorflow=={v['tensorflow']}")
    lines += ["python-pptx==1.0.2", "jupyter", "nbformat", "",
              "# 공휴일은 외부 패키지(holidays/workalendar) 대신 2021 상수 리스트를 사용한다",
              "# (폐쇄망 심사환경 대응)"]
    return "\n".join(lines) + "\n"


# ⚠️ 패키지 루트의 `requirements.txt` 를 덮어쓰지 않는다.
#    루트 파일은 사람이 관리하며 `ipykernel` 등 노트북 실행에 필요한 항목이 더 들어 있다.
#    여기서는 **현재 환경의 실제 설치 버전 증거**를 outputs/ 에 남기는 것이 목적이다.
#    (보고서 6장에서 "환경을 이렇게 고정했다"는 근거로 쓴다)
_req = write_requirements()
(OUTPUT_DIR / "requirements_generated.txt").write_text(_req, encoding="utf-8")
print("── outputs/requirements_generated.txt (실행 환경 증거) ──")
print(_req)


def write_readme() -> str:
    """README = 보고서 6장 본문."""
    nfig = len(list(FIG_DIR.glob("F*.png")))
    ntbl = len(list(TBL_DIR.glob("*.csv")))
    return f"""# 제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석

제6회 K-인공지능 제조데이터 분석 경진대회 — 과제 ⑤ 자원 최적화

## 1. 실행 방법

```bash
pip install -r requirements.txt
jupyter notebook 자원최적화_제안모델.ipynb
# Restart Kernel → Run All  (전체 실행 15분 내외)
```

전처리 → 학습 → 추론 → 결과생성이 자동으로 완료되며, `outputs/` 에
보고서용 표·그림·예측결과 파일이 모두 생성된다.
**모든 경로는 상대경로**이므로 zip 을 임의 위치에 풀어도 동작한다.

## 2. 파일 구성

```
자원최적화_제안모델.ipynb      메인 산출물 (0~10장)
data/okm_augumented_2021.csv   학습용 데이터 (6,168행 × 18열)
requirements.txt               의존성 (완전 핀)
src/s00~s10_*.py               노트북 소스 (단일 진실원, percent 포맷)
tools/build_notebook.py        src → ipynb 조립 스크립트
tests/                         단계별 검증 테스트
outputs/
  figures/     그림 {nfig}장 + figure_index.csv (그림↔보고서 절 매핑)
  tables/      표 {ntbl}개 (보고서 표 원본)
  predictions_test_336h.csv    테스트 예측결과 (336행)
  report_tbd_filled.md         채움표 + 문장 치환사전 + 장별 PASS/FAIL
  report_ch4_draft.md / ch5 / ch6   4·5·6장 본문 초안
```

## 3. 노트북 구조

| 장 | 내용 | 보고서 대응 |
|---|---|---|
| 0 | 실행 환경·재현성 설정 | 6장 |
| 1 | 데이터 이해 및 진단 | 1.2~1.4 |
| 2 | 파생변수 구성 및 시간누수 차단 | 1.5, 2.3 |
| 3 | 분할 설계 및 평가지표 | 1.6, 2.1 |
| 4 | 가이드북 베이스라인 재현 | 2.2 |
| 5 | 제안모델 개발 | 2.4, 2.5 |
| 6 | 성능평가 및 최종모델 선정 | 2.6~2.9 |
| 7 | 영향요인 및 오류분석 (입력=OOF) | 3장 |
| 8 | 피크 저감 시뮬레이션 (입력=전체기간 실측) | 4장 |
| 9 | 창의성·차별성 근거 | 5장 |
| 10 | 산출물·재현성·제출 패키징 | 6장 |

## 4. 재현성 보장 장치

| 장치 | 내용 |
|---|---|
| 시드 고정 | `PYTHONHASHSEED=42`, `random`/`numpy`/`keras` 시드 42 |
| 결정적 연산 | `TF_ENABLE_ONEDNN_OPTS=0`, `TF_DETERMINISTIC_OPS=1`, `enable_op_determinism()` |
| LightGBM | `deterministic=True`, `force_row_wise=True`, **`num_threads=4` 고정**(심사 PC 코어 수와 무관) |
| 의존성 | `requirements.txt` 를 `==` 로 완전 핀, 0.2절에서 버전 출력 |
| 한글 폰트 | 폴백 체인 (`Malgun Gothic` → `NanumGothic` → `Noto Sans KR`), 선택 폰트 출력 |
| 공휴일 | 외부 패키지 대신 2021 상수 리스트 (폐쇄망 대응) |
| TF 미설치 방어 | SimpleRNN·DNN 만 건너뛰고 파이프라인이 완주 |
| 무음 실패 승격 | 연쇄 대입을 `ChainedAssignmentError` 로 즉시 중단 |

현재 실행 환경: Python {env_versions().set_index('항목')['버전'].get('python')},
한글 폰트 `{FONT_NAME}`, TensorFlow {'사용' if HAS_TF else '미설치(건너뜀)'}.

## 5. 검증 장치 (게이트 5개 + 테스트)

노트북 내부에 단계별 게이트를 두어 **실패 시 즉시 중단**된다.

| 게이트 | 통과 조건 |
|---|---|
| 게이트 1 (1장) | (날짜,시간) 중복 0 · 시간 ∈[0,23] 100% · 결측 0 · 무결성 100% |
| 게이트 2 (2장) | 누수 검증 — 관측완료시각 ≤ 원점 **+ 미래맹검 동등성** |
| 게이트 3 (3장) | 사용 fold 복제 오염 0% |
| 게이트 4 (4장) | 무음 미반영 탐지 · RF 중요도 1변수 집중(버그 증명) |
| 게이트 5 (6장) | Naive 하한 통과 · 달력규칙 대비 FP 감소 |

`tests/` 의 pytest 스위트가 단계별 실측 오라클을 검증한다.

```bash
PYTHONHASHSEED=42 KAMP_FAST=1 python -X utf8 -m pytest tests/ -q
```

특히 **누수 탐지기 자체의 유효성**을 음성 대조로 증명한다 —
일부러 누수 피처(`rolling().shift(1)` 등)를 주입하면 탐지기가 반드시 실패한다.

## 6. 핵심 설계 결정

1. **`peak15 = max(15,30,45,60분)`** — 이 정의에서만 보고서의 187/201/222 가 재현된다.
2. **θ 고정 / τ 분리** — θ는 도메인 상수로 전 fold 고정, τ는 각 fold 검증구간에서만 산출.
   fold별 학습 q95 가 182~186 으로 어디서도 187이 안 나오므로 θ 고정이 필수다.
3. **origin = D일 24:00** — `D일 23:00` 으로 두면 `h=23`의 `lag24` 가 미완성 값을 참조한다.
4. **이동통계는 원점 기준 1회 계산 후 24시간 브로드캐스트** — `rolling().shift(1)` 금지.
5. **fold1 폐기** — 검증 14일 중 7일이 학습구간과 동일 전력 프로파일.
6. **7장 입력은 OOF, 8장 입력은 전체기간 실측** — 테스트는 피크 28건뿐이라 조건별 분석 불가.
"""


# ⚠️ 루트에 README.md 를 쓰지 않는다. 패키지의 진입점은 `00_README.md` 이며
#    사람이 관리한다. 여기서 생성하는 것은 **보고서 6장 본문 초안**이다.
_readme = write_readme()
(OUTPUT_DIR / "report_ch6_draft.md").write_text(
    "# 제 6 장. 코드 구성 및 재현성 〔10점〕\n\n" + _readme.split("# 제조 생산데이터")[-1].split("\n", 1)[1],
    encoding="utf-8",
)
print(f"── README.md ({len(_readme):,}자) · report_ch6_draft.md 생성 완료 ──")

# %% [markdown]
# ### 10.5 발표자료 골격 · 개인정보 스캔
#
# - **목적**: 발표자료 pptx 골격을 만들고, **제출 전 개인정보를 스캔**한다.
# - **보고서 대응절**: 6장, 제출물
# - **산출물**: `outputs/발표자료.pptx`, `ch6_privacy_scan.csv`
#
# 블라인드 평가이므로 소속·학교·로고 등 식별정보가 있으면 안 된다.
# **1건이라도 걸리면 빌드를 중단**해야 한다.
#
# > ### ⚠️ 이 셀이 볼 수 없는 것 — 자기 실행 출력
# >
# > `nbconvert --execute --inplace` 는 **모든 셀이 끝난 뒤에** 파일을 쓴다.
# > 따라서 이 셀이 읽는 노트북 파일에는 **아직 출력이 비어 있다.**
# > 서드파티 경고가 stderr 로 내보내는 설치 경로(사용자명 포함)는
# > 실행이 끝난 뒤에야 파일에 박히므로 여기서는 절대 잡을 수 없다.
# >
# > 그래서 제출 직전에 **반드시** 다음을 실행한다.
# >
# > ```bash
# > python build/finalize_notebook.py
# > ```
# >
# > 이 스크립트가 stderr 출력을 제거하고 잔여 경로를 치환한 뒤 후스캔한다.
# > (이 한계는 실제로 사용자명 6건이 출력에 남아 적발되면서 확인되었다)

# %%
BETA_STR = f"{BETA:.5f}"


def build_pptx() -> object:
    """발표자료 골격을 만든다 (핵심 그림 위주)."""
    try:
        from pptx import Presentation
        from pptx.util import Inches, Pt
    except Exception as e:  # pragma: no cover
        print(f"  python-pptx 미설치로 건너뜀 ({type(e).__name__})")
        return None

    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    def add_title_slide(title, subtitle):
        s = prs.slides.add_slide(prs.slide_layouts[0])
        s.shapes.title.text = title
        s.placeholders[1].text = subtitle
        return s

    def add_fig_slide(title, fid, bullet):
        s = prs.slides.add_slide(prs.slide_layouts[5])
        s.shapes.title.text = title
        hits = sorted(FIG_DIR.glob(f"{fid}_*.png"))
        if hits:
            s.shapes.add_picture(str(hits[0]), Inches(0.7), Inches(1.5), width=Inches(8.2))
        tb = s.shapes.add_textbox(Inches(9.2), Inches(1.6), Inches(3.6), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.text = bullet
        for p in tf.paragraphs:
            p.font.size = Pt(13)
        return s

    add_title_slide(
        "제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석",
        "제6회 K-인공지능 제조데이터 분석 경진대회 · 과제 ⑤ 자원 최적화",
    )
    add_fig_slide(
        "데이터의 44.7%가 합성 복제일",
        "F02",
        f"· 고유 프로파일 {N_UNIQUE_PROFILES}/{N_TOTAL_DAYS}일\n"
        f"· 복제 {N_SURPLUS_DAYS}일 (행 단위 중복 0건)\n"
        "· 전력은 동일, 기온은 -3.4~19.2°C\n"
        "· 오염 fold 폐기 → 성능 과대추정 차단",
    )
    add_fig_slide(
        "양방향 결측을 대칭 진단",
        "F04",
        "· 센서측 17시간 (평균전력 0)\n"
        "· ERP측 48시간 (생산량 0, 전력 정상)\n"
        "· 야간 피크 7건이 전부 ERP 결측일\n"
        "· 서로 다른 처리 규칙 적용",
    )
    add_fig_slide(
        "Day-ahead 누수 차단",
        "F13",
        "· origin = D일 24:00\n"
        "· 안전 lag 24/48/168 만\n"
        "· 이동통계는 원점 1회 계산 후 브로드캐스트\n"
        "· 미래맹검 동등성 검사로 기계적 증명",
    )
    add_fig_slide(
        "가이드북 베이스라인 결함 증명",
        "F16",
        "· 셀 73 `iloc[0]` → 실질 피처 1개\n"
        "· 학습 MSE > 테스트 MSE (과적합 0)\n"
        "· 무음 미반영 3곳 (Weekend/Vacation 전부 0)\n"
        "· 결함 13건을 표로 정리",
    )
    add_fig_slide(
        f"최종모델: {FINAL_MODEL_NAME}",
        "F17",
        f"· 테스트 MAE {FINAL_MAE:.2f} ({IMPROVE_MAE_PCT:+.1f}%)\n"
        f"· Recall {FINAL_RECALL:.3f} / F1 {FINAL_F1:.3f}\n"
        f"· 추론 {INFER_SEC:.2f}초 (336시간 배치)\n"
        f"· D2 순위상관 {float(rank_preservation.iloc[0]['값']):.3f}",
    )
    add_fig_slide(
        "달력규칙 기준선이 Recall 1.000",
        "F19",
        f"· 평일 ∧ 08~18시 → Recall 1.000, FP {CALENDAR_RULE_FP}\n"
        "· Recall 로는 어떤 모델도 이길 수 없다\n"
        "· 주지표를 F1/PR-AUC 로 재정의\n"
        "· 기여 = 동일 Recall 에서 FP 감소",
    )
    add_fig_slide(
        "피크 저감: 동시성 분산이 핵심",
        "F35",
        "· 08·13시만 분산 → Δ최대수요 0 kW\n"
        "· 09·11시까지 포함 → 13 kW\n"
        "· 청구 피크는 최대값 → 상위 사건 전부 덮어야\n"
        f"· 생산량 이동 β={BETA_STR} kW/단위 (지렛대 미미)",
    )
    add_fig_slide(
        "현장 운영 프로토콜",
        "F39",
        "· 전일 24:00 예측 → 24:10 조정안\n"
        "· 익일 07:30 계단식 기동 지시\n"
        "· 위험시간 실시간 감시\n"
        "· 규칙 R1~R3 과 연결된 점검 우선순위",
    )
    return prs


_prs = build_pptx()
if _prs is not None:
    _prs.save(str(OUTPUT_DIR / "발표자료.pptx"))
    print(f"── 발표자료.pptx 생성 ({len(_prs.slides._sldIdLst)}슬라이드) ──")


# ⚠️ 사용자명을 **소스에 literal 로 쓰지 않는다.**
# 여기에 이름을 적으면 그 이름이 노트북·스캔결과 CSV 에 그대로 실려
# 블라인드 평가 위반이 된다(실제로 그렇게 적발되었다).
# 실행 시점의 홈 디렉터리명에서 유도하면 어떤 심사 PC 에서도 동작한다.
from pathlib import Path as _Path  # noqa: E402

_USER = _Path.home().name

PRIVACY_PATTERNS = {
    "사용자명": re.escape(_USER),
    "Windows 절대경로": r"[Cc]:\+[Uu]sers",
    "홈 경로": r"/(?:home|Users)/[A-Za-z0-9_가-힣]+",
    "이메일": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "절대경로 출력": r"os\.getcwd\(\)",
}


def _mask(text: str) -> str:
    """발견된 문자열을 마스킹한다.

    스캔 결과 CSV 는 제출물에 포함되므로, 발견 내용을 원문 그대로 적으면
    스캐너가 오히려 개인정보를 유출한다. 첫 글자만 남기고 가린다.
    """
    text = text[:60]
    return text[:1] + "*" * max(len(text) - 1, 0)


ALLOWED_TEAM_NAME = "정종묵"   # 팀명·성명은 허용


# 제출 zip 에 포함되지 않는 개발 전용 문서 (스캔은 하되 차단 사유로 삼지 않는다)
DEV_ONLY = {"PRD.md", "CHECKLIST.md", "plan.md"}
# 스캐너 자신과 그 출력은 검사 대상에서 제외한다.
# 패턴 정의 문자열과 스캔 결과표가 매번 '발견'으로 잡히는 자기참조를 막는다.
SCANNER_SELF = {"s10_package.py", "ch6_privacy_scan.csv", "ch6_forbidden_scan.csv"}


def _scan_text(text: str, source: str, kind: str, rows: list) -> None:
    """한 덩어리 텍스트에 패턴을 적용해 결과를 rows 에 쌓는다."""
    for label, pat in PRIVACY_PATTERNS.items():
        # `os.getcwd()` 는 설명 문구에서도 등장하므로 코드에서만 위반으로 본다
        if label == "절대경로 출력" and kind != "code":
            continue
        for m in re.finditer(pat, text, re.MULTILINE):
            rows.append(
                {"파일": source, "위치": kind, "유형": label, "발견": _mask(m.group())}
            )


def _strip_comments(src: str) -> str:
    """파이썬 주석 줄을 제거한다 (설명 문구를 위반으로 오탐하지 않게)."""
    return chr(10).join(
        ln for ln in src.splitlines() if not ln.lstrip().startswith("#")
    )


def scan_privacy() -> pd.DataFrame:
    """제출 대상 파일에서 개인정보·절대경로를 스캔한다.

    노트북은 **JSON 원문을 그대로 훑지 않는다.** JSON 에서는 모든 소스 줄이
    따옴표로 감싸져 있어 주석 판별이 불가능하고, 이스케이프 때문에 패턴도
    어긋난다. `nbformat` 으로 파싱해 다음 세 곳을 구분해서 본다.

    - 코드 셀 소스 (주석 제거 후)
    - 마크다운 셀 소스 (설명이므로 PII 만 검사)
    - **셀 실행 출력** — 서드파티 경고에 설치 경로가 박혀 들어오는 주 경로다

    소스만 보면 실행 출력에 각인된 절대경로를 놓친다.
    """
    rows: list[dict] = []

    nb_path = PROJECT_ROOT / "자원최적화_제안모델.ipynb"
    if nb_path.exists():
        import nbformat

        nb = nbformat.read(nb_path, as_version=4)
        for i, cell in enumerate(nb.cells):
            src = cell.get("source", "") or ""
            if cell.cell_type == "code":
                _scan_text(_strip_comments(src), nb_path.name, "code", rows)
            else:
                _scan_text(src, nb_path.name, "markdown", rows)
            for out in cell.get("outputs", []) or []:
                chunks = []
                if "text" in out:
                    chunks.append(out["text"])
                for key in ("data",):
                    d = out.get(key) or {}
                    if isinstance(d, dict):
                        chunks.append(str(d.get("text/plain", "")))
                if out.get("output_type") == "error":
                    chunks.append(chr(10).join(out.get("traceback", []) or []))
                for c in chunks:
                    _scan_text(str(c), nb_path.name, "output", rows)

    targets = []
    targets += sorted(PROJECT_ROOT.glob("*.md"))
    targets += sorted(PROJECT_ROOT.glob("*.txt"))
    targets += sorted(OUTPUT_DIR.glob("*.md"))
    targets += sorted(TBL_DIR.glob("*.csv"))
    targets += sorted(PROJECT_ROOT.glob("src/*.py"))
    targets += sorted(PROJECT_ROOT.glob("tests/*.py"))

    for f in targets:
        if f.name in SCANNER_SELF:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        kind = "code" if f.suffix == ".py" else "doc"
        body = _strip_comments(text) if f.suffix == ".py" else text
        _scan_text(body, str(f.relative_to(PROJECT_ROOT)), kind, rows)

    out = pd.DataFrame(rows, columns=["파일", "위치", "유형", "발견"])
    if len(out):
        out["제출대상"] = ~(
            out["파일"].isin(DEV_ONLY) | out["파일"].str.startswith("tests")
        )
    else:
        out["제출대상"] = pd.Series(dtype=bool)
    return out


privacy_scan = scan_privacy()
save_table(privacy_scan, "ch6_privacy_scan")
# 제출 zip 에 실제로 들어가는 파일에서 발견된 것만 차단 사유로 삼는다.
# PRD.md·CHECKLIST.md·plan.md·tests/ 는 개발 전용이라 zip 에 포함하지 않는다.
privacy_blocking = (
    privacy_scan[privacy_scan["제출대상"]]
    if len(privacy_scan) and "제출대상" in privacy_scan.columns
    else privacy_scan.iloc[0:0]
)
print(
    f"\n── 개인정보 스캔: 전체 {len(privacy_scan)}건 / "
    f"**제출 대상 {len(privacy_blocking)}건** (개발 전용 문서는 zip 미포함) ──"
)
if len(privacy_scan) and {"유형", "파일"} <= set(privacy_scan.columns):
    display(privacy_scan.groupby(["유형", "파일", "위치", "제출대상"]).size().to_frame("건수").head(30))
    if len(privacy_blocking):
        print("\n  ⚠️ 제출 대상 파일에서 발견 — **zip 빌드 전에 반드시 제거**해야 한다.")
    else:
        print("\n  ✅ 제출 대상 파일에는 0건 (발견분은 전부 개발 전용 문서)")
else:
    print("  ✅ 개인정보·절대경로 0건")

# 노트북 내 금지 패턴 점검
# 문서·주석에 등장하는 '언급'이 아니라 **실제 호출**만 잡도록 좁힌다.
# (4.4절 결함표가 `freq='H'` 를 설명 문구로 담고 있어 단순 grep 은 오탐한다)
FORBIDDEN_IN_NOTEBOOK = {
    "이중 축 twinx()": r"^(?!\s*#).*\.twinx\(\)",
    "pandas 3.x 크래시 freq='H'": (
        r"^(?!\s*#).*(?:date_range|resample|period_range|Grouper)\s*\([^)]*"
        r"freq\s*=\s*['\"]H['\"]"
    ),
}


def scan_forbidden() -> pd.DataFrame:
    """노트북·stage 소스에서 금지 패턴을 찾는다."""
    rows = []
    for f in sorted(PROJECT_ROOT.glob("src/*.py")):
        if f.name in SCANNER_SELF:
            continue  # 패턴 정의 자체를 잡지 않는다
        text = f.read_text(encoding="utf-8")
        for label, pat in FORBIDDEN_IN_NOTEBOOK.items():
            hits = len(re.findall(pat, text, re.MULTILINE))
            if hits:
                rows.append({"파일": f.name, "금지 패턴": label, "건수": hits})
    return pd.DataFrame(rows, columns=["파일", "금지 패턴", "건수"])


forbidden_scan = scan_forbidden()
save_table(forbidden_scan, "ch6_forbidden_scan")
print(f"\n── 금지 패턴 스캔: {len(forbidden_scan)}건 ──")
if len(forbidden_scan):
    display(forbidden_scan)
else:
    print("  ✅ 이중 축(twinx)·freq='H' 0건")

# %% [markdown]
# ### 최종 게이트 — 제출 준비 상태 점검

# %%
# requirements.txt·README 가 10.4에서 생성되므로 장별 체크를 **여기서 다시** 돌린다.
# (10.2 시점에는 아직 파일이 없어 6장이 FAIL 로 잡힌다)
chapter_checks = check_chapters()
save_table(chapter_checks, "ch6_chapter_checklist")
_tbd_md = write_tbd_filled()
(OUTPUT_DIR / "report_tbd_filled.md").write_text(_tbd_md, encoding="utf-8")
print("── 장별 충족 체크리스트 (최종 재계산) ──")
display(chapter_checks.attrs["summary"].to_frame())


def final_gate() -> pd.DataFrame:
    """최종 게이트 — 제출 전 확인 항목."""
    nfig = len(list(FIG_DIR.glob("F*.png")))
    fig_idx = pd.read_csv(FIGURE_INDEX_PATH, encoding="utf-8-sig") if FIGURE_INDEX_PATH.exists() else pd.DataFrame()
    checks = [
        ("예측결과 336행·결측 0", len(predictions) == 336 and int(predictions.isna().sum().sum()) == 0),
        ("예측결과 9컬럼 규격", list(predictions.columns) == [
            "datetime", "y_avg_true", "y_avg_pred", "y_peak_true", "y_peak_pred",
            "peak_prob", "peak_pred_label", "peak_true_label", "model_name"]),
        ("그림 생성 수 == 인덱스 행 수", nfig == len(fig_idx)),
        ("장별 요건 전부 충족", bool(chapter_checks["충족"].all())),
        ("4·5·6장 본문 초안 존재", all(
            (OUTPUT_DIR / f"report_ch{i}_draft.md").exists() for i in (4, 5, 6))),
        ("requirements.txt 존재", (PROJECT_ROOT / "requirements.txt").exists()),
        ("진입 문서(00_README.md) 존재", (PROJECT_ROOT / "00_README.md").exists()),
        ("환경 증거 생성", (OUTPUT_DIR / "requirements_generated.txt").exists()),
        ("이중 축·freq='H' 0건", len(forbidden_scan) == 0),
        ("제출 대상 개인정보 0건", len(privacy_blocking) == 0),
        ("채움표 30항목 이상", len(tbd_tbl) >= 30),
    ]
    out = pd.DataFrame(checks, columns=["점검", "통과"])
    return out


gate_final = final_gate()
save_table(gate_final, "gate6_final")
print("── 최종 게이트 ──")
display(gate_final)
_nf = gate_final[~gate_final["통과"]] if "통과" in gate_final.columns else gate_final.iloc[0:0]
if len(_nf):
    print(f"\n  ⚠️ 미통과: {_nf['점검'].tolist()}")
else:
    print("\n  ✅ 최종 게이트 전 항목 통과")

print(f"\n{'=' * 64}")
print("제출 준비 요약")
print("=" * 64)
print(f"  최종모델        : {FINAL_MODEL_NAME}")
print(f"  테스트 MAE      : {FINAL_MAE:.3f} ({IMPROVE_MAE_PCT:+.2f}% vs {BEST_BASELINE_NAME})")
print(f"  Recall / F1     : {FINAL_RECALL:.3f} / {FINAL_F1:.3f}")
print(f"  최대수요 저감   : {PEAK_REDUCTION_KW:.1f} kW (전체기간 기준)")
print(f"  그림            : {len(list(FIG_DIR.glob('F*.png')))}장")
print(f"  표              : {len(list(TBL_DIR.glob('*.csv')))}개")
print(f"  FAST 모드       : {FAST}  {'← 제출 전 FULL 재실행 필요' if FAST else ''}")
print("=" * 64)
