# %% tags=["nb-strip"]
from s00_env import COLOR_HERO, INK, OUTPUT_DIR, display, save_table  # noqa: F401
from s01_diagnose import (  # noqa: F401
    N_DUP_GROUPS,
    N_SURPLUS_DAYS,
    N_TOTAL_DAYS,
    N_UNIQUE_PROFILES,
    THETA,
    df,
)
from s03_split import CALENDAR_RULE_FP, calendar_rule_metrics  # noqa: F401
from s05_models import calibration_tbl, ensemble_info, uncertainty_tbl  # noqa: F401
from s06_eval import (  # noqa: F401
    BEST_BASELINE_NAME,
    FINAL_MODEL_NAME,
    FINAL_MAE,
    FINAL_RECALL,
    IMPROVE_MAE_PCT,
    MAE_IMPROVEMENT_SIGNIFICANT,
    ablation_tbl,
    interpretation_tbl,
    peak_tbl,
    rank_preservation,
)
from s07_analysis import TOP_FEATURES, WORST_CONDITION_NAME, WORST_COND_EXCESS_PCT, rules_tbl  # noqa: F401
from s08_simulation import (  # noqa: F401
    BETA,
    PEAK_REDUCTION_KW,
    SEPT_ONLY_SAVING,
    levers_tbl,
    scenarios_tbl,
)
import numpy as np
import pandas as pd

# %% [markdown]
# ## 9. 창의성·차별성 근거 산출
#
# 보고서 **5장 창의성 및 차별성(10점)** 의 근거를 생성한다.
#
# > 채점 기준은 "새로운 알고리즘을 썼다"가 아니라
# > **"제안한 방법이 성능·오류 감소·현장 활용에 어떤 기여를 했는지를 2·3장 결과에 연결"** 하는 것이다.
# > 따라서 이 장은 새 실험을 하지 않고, 앞 장에서 **이미 측정된 수치**를 차별점으로 재구성한다.

# %% [markdown]
# ### 9.1 ~ 9.5 차별점 5종 정리
#
# - **목적**: 각 차별점을 "무엇을 했는가 → 어떤 수치로 확인되는가 → 어디에 기여하는가"로 정리한다.
# - **보고서 대응절**: 5장
# - **산출물**: `ch5_creativity.csv`

# %%
def build_creativity_table() -> pd.DataFrame:
    """5장 차별점 5종을 앞 장의 실측 수치와 연결한다."""
    lag_row = ablation_tbl[ablation_tbl["제거 피처군"] == "과거 전력 지연변수"]
    regime_row = ablation_tbl[ablation_tbl["제거 피처군"] == "레짐분리"]
    final_fp = int(peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME]["FP"].iloc[0])
    calib = calibration_tbl.set_index("구분")
    l1a = levers_tbl[levers_tbl["레버"].str.startswith("L1a")].iloc[0]
    l1c = levers_tbl[levers_tbl["레버"].str.startswith("L1c")].iloc[0]

    rows = [
        {
            "차별점": "① 증강 복제 구조 진단",
            "무엇을 했는가": (
                "일 단위 전력 프로파일 해시로 데이터의 구조적 복제를 탐지했다. "
                "행 단위 완전중복이 0건이라 일반 `duplicated()` 로는 전혀 잡히지 않는다."
            ),
            "측정된 근거": (
                f"고유 프로파일 {N_UNIQUE_PROFILES}/{N_TOTAL_DAYS}일, 중복그룹 {N_DUP_GROUPS}개, "
                f"복제 {N_SURPLUS_DAYS}일({N_SURPLUS_DAYS / N_TOTAL_DAYS:.1%}). "
                "최대 그룹 15일은 전력이 동일한데 기온은 -3.4~19.2°C. "
                "추가로 생산량 0인데 전력이 정상 수준인 '채널 불일치' 9일을 발견."
            ),
            "기여": (
                "오염된 fold(검증 50%가 학습과 동일 프로파일)를 폐기해 성능 과대추정을 차단했고, "
                f"복제 제거 조건 D2에서 순위상관 {float(rank_preservation.iloc[0]['값']):.3f}로 "
                "결론의 강건성을 확인했다."
            ),
            "대응 절": "1.2, 1.4, 2.6, 2.9",
        },
        {
            "차별점": "② 불균형 학습 + 판정 임계값 분리",
            "무엇을 했는가": (
                "피크 정의 임계값 θ와 경보 판정 임계값 τ를 **기호부터 분리**했다. "
                "θ는 도메인 상수로 전 fold 고정, τ는 각 fold 검증구간에서만 산출한다."
            ),
            "측정된 근거": (
                "fold별 학습구간 q95가 182/182/184/186/186으로 **어느 fold에서도 187이 나오지 않는다** "
                "→ θ를 fold마다 재산출하면 y_cls 정의 자체가 달라져 fold 평균이 무의미해진다. "
                "회귀 예측은 평균으로 수축하므로 θ 고정 적용 시 Recall이 구조적으로 낮아진다."
            ),
            "기여": (
                "2.7절 표에 각 행이 쓴 τ를 컬럼으로 노출해 비교 가능성을 확보했고, "
                "class_weight 로 양성 비중을 확대했다."
            ),
            "대응 절": "1.6, 2.5, 2.7",
        },
        {
            "차별점": "③ 평가지표 재정의 — 달력규칙 기준선 도입",
            "무엇을 했는가": (
                "`평일 ∧ 08≤h≤18` 한 줄 규칙을 2.7절 표의 **기준 행**으로 추가했다."
            ),
            "측정된 근거": (
                f"달력규칙이 테스트에서 Recall 1.000 (TP 28 / FN 0 / FP {CALENDAR_RULE_FP} / TN 226)을 낸다. "
                "**어떤 ML 모델도 Recall 로는 이길 수 없다.**"
            ),
            "기여": (
                f"주지표를 'Recall 유지 + 오경보 최소화'(F1/PR-AUC)로 재정의하고, 기여를 "
                f"**동일 수준에서 FP {CALENDAR_RULE_FP}건 → {final_fp}건 "
                f"({CALENDAR_RULE_FP - final_fp}건 감소)** 로 정량화했다. "
                "기준선이 없으면 'Recall 0.9 달성'이 실제보다 대단해 보이는 착시가 생긴다."
            ),
            "대응 절": "2.1, 2.7",
        },
        {
            "차별점": "④ 확률 보정 + 예측 불확실성",
            "무엇을 했는가": (
                "Isotonic 회귀로 피크 확률을 보정하고, 분위회귀와 Split Conformal 로 예측구간을 제시했다."
            ),
            "측정된 근거": (
                f"Brier {calib.loc['보정 전', 'Brier']:.5f} → {calib.loc['보정 후', 'Brier']:.5f}, "
                f"ECE {calib.loc['보정 전', 'ECE']:.5f} → {calib.loc['보정 후', 'ECE']:.5f}. "
                "예측구간 피복률을 실측으로 보고했고, 보정집합에 하계휴가가 포함되면 "
                "구간이 크게 넓어지는 분포이동 효과까지 드러냈다."
            ),
            "기여": (
                "경보를 확률 기준으로 운영할 수 있게 되어 '확률 0.7 = 실제 70% 발생'이 성립한다. "
                "예측구간은 위험시간의 감시 강도를 정하는 근거가 된다."
            ),
            "대응 절": "5장, 4.6",
        },
        {
            "차별점": "⑤ 제조지식 결합 — 레짐 분리와 동시성 분산",
            "무엇을 했는가": (
                "가동/비가동 이중 구조를 2단계 레짐 모델로 명시하고, "
                "피크 저감을 '물량 재배치'가 아니라 '동시 가동 분산'으로 설계했다."
            ),
            "측정된 근거": (
                f"레짐분리를 제거하면 OOF MAE가 {float(regime_row['MAE 변화'].iloc[0]):+.3f} kW 악화되어 "
                "전체 피처군 중 기여가 가장 크다. "
                f"pooled 반응함수 β={BETA:.5f} kW/생산1단위로 생산량 이동의 지렛대는 미미하다. "
                f"반면 기동 분산은 08·13시만 덮으면 Δ최대수요 {l1a['Δ최대수요(kW)']:.1f} kW, "
                f"09·11시까지 덮으면 {l1c['Δ최대수요(kW)']:.1f} kW 를 달성한다."
            ),
            "기여": (
                "생산 일정을 크게 흔들지 않고 피크를 관리할 수 있음을 수치로 보였다. "
                "청구 피크가 최대값이므로 **상위 사건을 순차적으로 모두 덮어야** 절감이 생긴다는 "
                "설계 원칙도 도출했다."
            ),
            "대응 절": "2.4, 2.8, 4장",
        },
    ]
    return pd.DataFrame(rows)


creativity_tbl = build_creativity_table()
save_table(creativity_tbl, "ch5_creativity")
print("── 5장 차별점 5종 ──")
for _, r in creativity_tbl.iterrows():
    print(f"\n[{r['차별점']}] (보고서 {r['대응 절']})")
    print(f"  무엇: {r['무엇을 했는가']}")
    print(f"  근거: {r['측정된 근거']}")
    print(f"  기여: {r['기여']}")

# %% [markdown]
# ### 9.6 보고서 5장 본문 초안 생성
#
# - **목적**: 차별점 5종을 5장 본문 초안 markdown 으로 출력한다.
# - **보고서 대응절**: 5장 전체
# - **산출물**: `outputs/report_ch5_draft.md`

# %%
def write_ch5_draft() -> str:
    """5장 '창의성 및 차별성' 본문 초안을 생성한다."""
    final_fp = int(peak_tbl[peak_tbl["모델"] == FINAL_MODEL_NAME]["FP"].iloc[0])
    md = f"""# 제 5 장. 창의성 및 차별성 〔10점〕

본 장은 새로운 알고리즘을 사용했다는 설명이 아니라, 제안한 방법이 **제2장(성능)과
제3장(오류분석)의 결과에 어떻게 기여했는지**를 측정된 수치로 연결한다.

"""
    for i, (_, r) in enumerate(creativity_tbl.iterrows(), start=1):
        md += f"""## 5.{i} {r['차별점'][2:]}

**무엇을 했는가.** {r['무엇을 했는가']}

**측정된 근거.** {r['측정된 근거']}

**기여.** {r['기여']} (보고서 {r['대응 절']}절)

"""
    md += f"""## 5.6 정직한 한계 보고

차별성은 성과를 부풀리는 데 있지 않고 **결과의 해석 가능성을 높이는 데** 있다고 보았다.
따라서 다음 한계를 수치와 함께 명시했다.

1. **MAE 개선의 통계적 유의성.** 최종모델의 MAE 개선 {IMPROVE_MAE_PCT:+.2f}%는
   점추정으로는 개선이지만, 일 단위 블록 부트스트랩 95% 신뢰구간이
   {'0을 포함하지 않아 유의하다' if MAE_IMPROVEMENT_SIGNIFICANT else '0을 포함해 **통계적으로 유의하지 않다**'}.
   테스트 336시간은 일 블록이 14개뿐이라 검정력이 낮다. 그래서 선정 근거를
   MAE 단독이 아니라 Peak-MAE·오경보 감소·폴드 편차로 분산했다.

2. **테스트 구간의 소표본 문제.** 테스트 피크는 28건이며 8개 날짜·16개 연속블록에만
   분포한다. Recall 0.357 기준 Clopper-Pearson 95% CI 폭이 0.37이므로
   테스트만으로는 두 모델을 구분할 수 없다. 이 때문에 모델 비교의 주 근거를
   Rolling-origin OOF(양성 156건)로 두고, 테스트는 최종 확인용 단일 수치로 사용했다.

3. **9월 구간 단독으로는 기본요금 절감이 0원.** 요금이 3개월 최대피크 기준이고
   7~9월 창의 최대는 7월(222 kW)이므로, 9월을 0으로 낮추는 극단 가정에서도
   연간 기본요금 절감액은 {SEPT_ONLY_SAVING:,.0f}원이다. 이를 숨기지 않고 명시하고,
   시뮬레이션 대상을 전체기간 실측으로 확대했다.

4. **지연변수의 조건부 유용성.** OOF 구간에 하계휴가가 포함되어 `lag168`의 참조 시각이
   휴무일이 되는 구간에서는 지연변수가 오히려 해롭다. 2.8절에서 이를 수치로 보이고,
   향후 개선방향으로 **휴무 인지형 지연변수**를 제시했다.

5. **가정의 명시와 민감도.** 기본요금 단가와 α(동시 기동 기여분)는 제공 데이터로
   검증할 수 없으므로 파라미터로 노출하고 ±30% 민감도를 제시했으며,
   시나리오 표에 '모델기반 / 가정기반' 열을 두어 근거 강도를 구분했다.

이처럼 **무엇을 확인했고 무엇을 확인할 수 없었는지를 분리해 보고한 것**이
본 연구의 가장 실질적인 차별점이다.
"""
    return md


_ch5 = write_ch5_draft()
(OUTPUT_DIR / "report_ch5_draft.md").write_text(_ch5, encoding="utf-8")
print(f"\n── 5장 본문 초안 생성 완료 ({len(_ch5):,}자) → outputs/report_ch5_draft.md ──")
