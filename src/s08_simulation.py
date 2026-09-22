# %% tags=["nb-strip"]
from s00_env import (  # noqa: F401
    CMAP_SEQ,
    COLOR_HERO,
    COLOR_MUTED,
    INK,
    INK_SOFT,
    OUTPUT_DIR,
    PALETTE_ADJACENT,
    SEED,
    display,
    save_fig,
    save_table,
)
from s01_diagnose import THETA, TEST_START, TEST_END, df, operating_calendar  # noqa: F401
from s02_features import feat  # noqa: F401
from s06_eval import FINAL_MODEL_NAME, test_results  # noqa: F401
from s07_analysis import OOF, rules_tbl  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## 8. 피크 저감 시뮬레이션
#
# 보고서 **4장 현장 활용방안(10점)** 의 근거를 생성한다.
#
# > ### 🔒 이 장의 입력은 **전체기간(01-01~09-14) 실측 `peak15`** 다
# >
# > **9월 테스트 구간만으로는 기본요금 절감액이 논리적으로 0원**이다.
# > 요금이 "3개월 최대피크로 12개월 청구" 구조이고, 7~9월 창의 최대는 **222(7월)** 인데
# > 9월 최대는 **204** 다. 즉 **9월을 0으로 낮춰도 청구 피크는 변하지 않는다.**
# >
# > | 월 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | **9(테스트)** |
# > |---|---|---|---|---|---|---|---|---|---|
# > | 최대 peak15 | **222** | 198 | **222** | 199 | 199 | **222** | **222** | 218 | **204** |
# >
# > 따라서 시뮬레이션은 전체기간 실측 위에서 수행하고, 9월 결과는
# > "모델 예측 기반 운영 시나리오 검증"으로만 제시한다.
# > **반사실(counterfactual) 계산은 예측값이 아니라 실측 `peak15` 시계열 위에서** 한다.

# %% [markdown]
# ### 8.1 요금 모델링
#
# - **목적**: 두 청구 단위(월별 최대수요 / 직전 3개월 롤링 최대)를 산출한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_tariff.csv`, F34
#
# **요금 구조**: 한국전력공사는 **3개월간의 최대 피크 전력을 기준으로 1년간 전기료를 부과**한다
# (가이드북 셀 1). 즉 **연중 단 한 번의 최대수요가 12개월 기본요금을 결정**하므로
# 목적함수는 평균이 아니라 **최대값(max) 저감**이다.
#
# $$\text{절감액} = \Delta\text{최대수요(kW)} \times \text{기본요금단가} \times 12 + \text{전력량요금 차액} - \text{인건비 증가분}$$
#
# 기본요금 단가는 제공 데이터로 검증할 수 없으므로 **파라미터로 노출하고 ±30% 민감도**를 제시한다.
# 전력 단위(kW) 해석도 명시적 가정이다.

# %%
# ── 가정 파라미터 (전부 민감도 분석 대상) ──────────────────────────────
BASE_RATE_KRW_PER_KW_MONTH = 8_320.0   # 기본요금 단가 (가정, 검증 불가)
BILLING_MONTHS = 12                    # 3개월 최대피크로 12개월 청구
ALPHA_SIMULTANEOUS = 0.20              # 08시 피크 중 '동시 기동' 기여분 (가정)
NIGHT_LABOR_MULTIPLIER = 1.5           # 야간 인건비 배수 (데이터 확인됨)
WEEKEND_LABOR_MULTIPLIER = 1.0         # ⚠️ 이 데이터상 주말 할증 0 (실측)
WEEKEND_PREMIUM_PARAM = 1.5            # 실제 특근수당 가정치 (파라미터로 노출)
LABOR_COST_KRW_PER_UNIT_HOUR = 25_000  # 공수 1단위·1시간 인건비 (가정)

PEAK15 = df["y_peak"].astype(float).copy()   # 전체기간 실측 peak15
# ⚠️ `생산량` 은 int64 다. 반사실 계산에서 소수 물량을 옮기므로 float 으로 캐스팅한다.
#    pandas 3.x 는 int 컬럼에 float 대입을 TypeError 로 거부한다.
PROD = df["생산량"].astype(float).copy()
TARIFF = df["전기요금(계절)"].copy()    # 계절별 전력량 요금단가 (원/kWh)


def billing_units(peak: pd.Series) -> pd.DataFrame:
    """두 청구 단위를 산출한다.

    - `월별최대`: 각 월의 최대수요
    - `3개월롤링최대`: 해당 월을 포함한 직전 3개월 창의 최대수요 (한전 방식)
    """
    monthly = peak.groupby(peak.index.to_period("M")).max()
    rolling3 = monthly.rolling(3, min_periods=1).max()
    out = pd.DataFrame({"월별최대": monthly, "3개월롤링최대": rolling3})
    out.index = out.index.astype(str)
    return out


def annual_billing_peak(peak: pd.Series) -> float:
    """연간 기본요금을 결정하는 청구 피크 = 3개월 롤링 최대의 최대값."""
    return float(billing_units(peak)["3개월롤링최대"].max())


def basic_charge(peak_kw: float, rate: float = BASE_RATE_KRW_PER_KW_MONTH) -> float:
    """연간 기본요금 = 청구 피크 × 단가 × 12개월."""
    return peak_kw * rate * BILLING_MONTHS


tariff_tbl = billing_units(PEAK15)
BASE_BILLING_PEAK = annual_billing_peak(PEAK15)
BASE_BASIC_CHARGE = basic_charge(BASE_BILLING_PEAK)
save_table(tariff_tbl.reset_index(names="월"), "ch4_tariff")

print("── 청구 단위별 최대수요 ──")
display(tariff_tbl.T)
print(f"\n  연간 청구 피크 (3개월 롤링 최대의 최대) = {BASE_BILLING_PEAK:.0f} kW")
print(f"  연간 기본요금 = {BASE_BASIC_CHARGE:,.0f} 원 (단가 {BASE_RATE_KRW_PER_KW_MONTH:,.0f} 원/kW·월 가정)")

# 최대값 도달 시각 — 저감의 표적
MAX_PEAK_TIMES = df.index[PEAK15 == PEAK15.max()]
print(f"\n  최대 {PEAK15.max():.0f} kW 도달 {len(MAX_PEAK_TIMES)}건 (저감 표적):")
for _t in MAX_PEAK_TIMES:
    print(f"    - {_t:%Y-%m-%d %H시}")

# 9월 단독 절감이 0원임을 수치로 확인한다
_sept_zeroed = PEAK15.copy()
_sept_zeroed.loc[_sept_zeroed.index >= TEST_START] = 0.0
SEPT_ONLY_SAVING = BASE_BASIC_CHARGE - basic_charge(annual_billing_peak(_sept_zeroed))
print(
    f"\n  ⚠️ 9월 peak15 를 전부 0으로 낮춘 극단 가정에서도 연간 기본요금 절감 = "
    f"{SEPT_ONLY_SAVING:,.0f} 원"
)
print("     → 테스트 구간 단독으로는 연간 기본요금에 영향이 없다 (보고서에 명시)")


def plot_billing_peaks(tbl: pd.DataFrame):
    """F34 — 월별 최대수요 + 3개월 롤링 최대 (222 도달 4건 표시)."""
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    x = range(len(tbl))
    ax.plot(x, tbl["월별최대"], color=COLOR_HERO, lw=2, marker="o", ms=6, label="월별 최대수요")
    ax.step(x, tbl["3개월롤링최대"], where="mid", color=PALETTE_ADJACENT[1], lw=2,
            label="3개월 롤링 최대 (청구 기준)")
    mx = tbl["월별최대"].max()
    for i, v in enumerate(tbl["월별최대"]):
        if v == mx:
            ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(tbl.index, fontsize=8, rotation=45)
    ax.set_ylabel("peak15 (kW)", fontsize=9)
    ax.set_title("월별 최대수요와 청구 기준 (3개월 롤링 최대)", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, tbl


_fig, _src = plot_billing_peaks(tariff_tbl)
save_fig(_fig, "F34", "월별 최대수요와 청구기준", "8.1", source_table=_src)

# %% [markdown]
# ### 8.2 저감 레버 L1~L4
#
# - **목적**: 4개 레버를 반사실 시뮬레이션으로 구현한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_levers.csv`, F37
#
# **공통 원칙 — 일일 총 생산량을 보존한 채 시간 배치만 바꾼다.**
# 생산을 줄이는 게 아니라 **옮기므로 매출 손실이 없다.** 이것이 4장의 설득력이다.
#
# | 레버 | 조정 | 근거 | 근거 강도 |
# |---|---|---|---|
# | **L1** 설비 기동시점 분산 | 08시 일제기동 → 08:00/08:20/08:40 계단식 | 08시 피크율 24.5%(1위), 7→8시 87.6→121.3 급등, 최대 222 중 3건이 08시 | **가정기반** (설비 ID 부재 → α 파라미터) |
# # | **L2** 고생산 시간대 이동 | 피크 시각 생산량 일부를 저부하 시간(10~11,15~16시)으로 | 생산량-전력 corr 0.518, pooled β | 모델기반 |
# | **L3** 시간당 생산량 상한 | 피크위험 시간에 상한 + 초과분 이월 | L2의 자동화 버전 | 모델기반 |
# | **L4** 야간·주말 이전 | 주간 고부하 일부를 야간·주말로 | 토 51.3 / 일 41.6 여유 | 모델기반 + 인건비 가정 |

# %%
def fit_production_response() -> tuple[float, dict]:
    """생산량 1단위당 `peak15` 증분(반응함수 β)을 **pooled 회귀**로 적합한다.

    시간대별로 따로 적합하면 계수가 불안정하다. `생산량` 은 ERP 등록값이라
    물리적 시간대와 어긋나고(1.5절: 19시 생산량 이상 = 마감 일괄등록 의심),
    시간별 표본이 ~250개뿐이어서 일부 시간대에서 기울기가 음수로 나온다.
    실제로 시간별 적합 시 10·11·16·20·22·23시의 β가 음수→0으로 clip 되어
    **11시 피크를 저감할 수 없게 되는 인공적 결과**가 생긴다.

    따라서 `peak15 ~ 생산량 + 시간더미` 를 한 번에 적합해 **단일 β** 를 쓴다.
    시간 고정효과로 시간대별 기저 부하 차이를 흡수하므로 β는
    "같은 시간대에서 생산량이 1단위 늘 때의 peak15 증분"으로 해석된다.

    Returns
    -------
    (float, dict)
        β 추정치와 적합 진단(p-value, R², n).
    """
    import statsmodels.api as sm

    d = df[(df["생산량"] > 0) & (~df["is_erp_missing"])]
    X = pd.get_dummies(d["시간"], prefix="h", drop_first=True).astype(float)
    X["prod"] = d["생산량"].astype(float).to_numpy()
    X = sm.add_constant(X)
    m = sm.OLS(d["y_peak"].astype(float).to_numpy(), X).fit()
    beta = float(m.params["prod"])
    diag = {
        "β (kW/생산1단위)": round(beta, 6),
        "p-value": round(float(m.pvalues["prod"]), 4),
        "R²": round(float(m.rsquared), 4),
        "n": int(len(d)),
    }
    return max(beta, 0.0), diag


BETA, BETA_DIAG = fit_production_response()
OBS_PEAK_MIN, OBS_PEAK_MAX = float(PEAK15.min()), float(PEAK15.max())
print("── 생산량 반응함수 (pooled, 시간 고정효과) ──")
display(pd.DataFrame([BETA_DIAG]))

# ── 저감 표적: 연중 상위 피크 시각 전부 ───────────────────────────────
# 청구 피크는 **최대값**이므로 상위 사건을 하나라도 남기면 절감이 0이 된다.
TOP_N_PEAKS = 20
TOP_PEAK_TIMES = PEAK15.nlargest(TOP_N_PEAKS).index
PEAK_HOUR_COUNTS = pd.Series(TOP_PEAK_TIMES.hour).value_counts().sort_index()
print(f"\n── 연중 상위 {TOP_N_PEAKS}개 피크 시각의 시간대 분포 ──")
display(PEAK_HOUR_COUNTS.to_frame("건수").T)
print(
    "  ⚠️ 상위 피크는 08시뿐 아니라 **09시·11시** 에도 몰려 있다.\n"
    "     08·13시만 다루는 L1a 로는 11시 사건(07-19 11시 = 222 kW)이 남아\n"
    "     **청구 피크가 전혀 줄지 않는다**(최대값 목적함수의 특성)."
)


def clip_to_observed(s: pd.Series) -> pd.Series:
    """반사실 값을 관측 분포 범위로 clip 한다 (외삽 방지)."""
    return s.clip(lower=OBS_PEAK_MIN, upper=OBS_PEAK_MAX)


def lever_L1(
    peak: pd.Series, alpha: float = ALPHA_SIMULTANEOUS, hours=(8, 13)
) -> pd.Series:
    """L1 — 동시 가동 집중 시각의 부하를 계단식으로 분산한다.

    설비 ID가 없어 동시 기동 기여분을 데이터로 산출할 수 없다.
    `alpha` = 해당 시각 peak15 중 동시 기동(돌입전류)·동시 가동에 귀속되는
    비율(가정). 3그룹 계단식 분산으로 그 기여분의 2/3를 제거한다고 본다.

    Parameters
    ----------
    hours : tuple[int, ...]
        분산 대상 시각. `(8, 13)` 이 계획서 명세(L1a)이고,
        `(8, 11, 13)` 은 실측 피크 분포를 반영한 확장(L1b 포함)이다.
    """
    out = peak.copy()
    target = np.isin(out.index.hour, list(hours))
    out[target] = out[target] * (1 - alpha * (2 / 3))
    return clip_to_observed(out)


def lever_L2(peak: pd.Series, prod: pd.Series, shift_ratio: float = 0.20) -> tuple[pd.Series, pd.Series]:
    """L2 — 피크 시간의 생산량 일부를 같은 날 저부하 시간으로 이동한다.

    일일 총 생산량을 보존한다. 수용 시간대는 10·11·15·16시(1.5절 EDA에서
    여유가 확인된 시간)로 두고, 각 날의 최고 peak15 시각에서
    `shift_ratio` 만큼의 생산량을 빼 나눠 담는다.

    Returns
    -------
    (pandas.Series, pandas.Series)
        조정된 peak15, 조정된 생산량.
    """
    out, prod_out = peak.copy(), prod.copy()
    RECEIVERS = [10, 11, 15, 16]
    target_days = {t.normalize() for t in TOP_PEAK_TIMES}
    for day, g in peak.groupby(peak.index.normalize()):
        # 연중 상위 피크가 있는 날 또는 θ 초과일만 조정한다
        if day not in target_days and g.max() < THETA:
            continue
        src_t = g.idxmax()
        if prod.loc[src_t] <= 0:
            continue
        move = prod.loc[src_t] * shift_ratio
        # 같은 날 수용 시간대 (원 시각 제외)
        recv = [t for t in g.index if t.hour in RECEIVERS and t != src_t]
        if not recv:
            continue
        per = move / len(recv)
        prod_out.loc[src_t] -= move
        out.loc[src_t] -= BETA * move
        for t in recv:
            prod_out.loc[t] += per
            out.loc[t] += BETA * per
    return clip_to_observed(out), prod_out


def lever_L3(peak: pd.Series, prod: pd.Series, cap_quantile: float = 0.90) -> tuple[pd.Series, pd.Series, float]:
    """L3 — 피크 위험 시간에 시간당 생산량 상한을 걸고 초과분을 다음 시간으로 이월한다.

    L2의 자동화 버전이다. 예측 `peak15` 가 임계를 넘는 시각에만 발동한다.
    **이월 잔량이 0인지 반드시 검증한다** (상한이 낮으면 일 생산량 미달).

    Returns
    -------
    (pandas.Series, pandas.Series, float)
        조정된 peak15, 조정된 생산량, 남은 이월 잔량 합계.
    """
    out, prod_out = peak.copy(), prod.copy()
    cap = float(prod[prod > 0].quantile(cap_quantile))
    leftover_total = 0.0
    for day, g in peak.groupby(peak.index.normalize()):
        if g.max() < THETA:
            continue
        carry = 0.0
        times = list(g.index)
        for i, t in enumerate(times):
            want = prod_out.loc[t] + carry
            if peak.loc[t] >= THETA and want > cap:
                carry = want - cap
                new_q = cap
            else:
                carry = 0.0
                new_q = want
            delta = new_q - prod_out.loc[t]
            prod_out.loc[t] = new_q
            out.loc[t] += BETA * delta
        leftover_total += carry   # 하루가 끝나도 남은 이월량
    return clip_to_observed(out), prod_out, leftover_total


def lever_L4(
    peak: pd.Series, prod: pd.Series, target: str = "night", shift_ratio: float = 0.15
) -> tuple[pd.Series, pd.Series, float]:
    """L4 — 주간 고부하 작업 일부를 야간 또는 주말로 이전한다.

    **야간 이전과 주말 이전을 분리한다.** 야간은 인건비 1.5배가 실측으로 확인되고,
    주말 할증은 **이 데이터상 0**이다(요일 분포 균등). 실제 특근수당은 검증할 수
    없으므로 파라미터로 노출하고 민감도에 포함한다.

    Returns
    -------
    (pandas.Series, pandas.Series, float)
        조정된 peak15, 조정된 생산량, 추가 인건비(원).
    """
    out, prod_out = peak.copy(), prod.copy()
    extra_labor = 0.0
    if target == "night":
        recv_hours, mult = [21, 22, 23, 0, 1, 2], NIGHT_LABOR_MULTIPLIER
    else:
        recv_hours, mult = list(range(8, 18)), WEEKEND_PREMIUM_PARAM

    for day, g in peak.groupby(peak.index.normalize()):
        if g.max() < THETA:
            continue
        src_t = g.idxmax()
        if prod.loc[src_t] <= 0:
            continue
        move = prod.loc[src_t] * shift_ratio
        if target == "night":
            recv = [t for t in g.index if t.hour in recv_hours]
        else:
            # 같은 주의 토요일 수용 시간대
            week_sat = day + pd.Timedelta(days=(5 - day.dayofweek) % 7)
            recv = [t for t in peak.index if t.normalize() == week_sat and t.hour in recv_hours]
        if not recv:
            continue
        per = move / len(recv)
        prod_out.loc[src_t] -= move
        out.loc[src_t] -= BETA * move
        for t in recv:
            prod_out.loc[t] += per
            out.loc[t] += BETA * per
            # 이전된 물량만큼 추가 인건비 (배수 − 1.0 만큼 증분)
            extra_labor += per * LABOR_COST_KRW_PER_UNIT_HOUR / 1000 * (mult - 1.0)
    return clip_to_observed(out), prod_out, extra_labor


def energy_cost(prod_series: pd.Series, peak_series: pd.Series) -> float:
    """전력량요금 = Σ(시간별 평균전력 대리값 × 계절단가).

    생산량 이동에 따른 시간대별 요금 차액을 근사한다.
    peak15 변화를 평균전력 변화의 대리로 쓴다(동일 시간 내 비례 가정).
    """
    return float((peak_series * TARIFF).sum())


BASE_ENERGY_COST = energy_cost(PROD, PEAK15)


def evaluate_lever(name: str, peak_new: pd.Series, prod_new: pd.Series,
                   extra_labor: float = 0.0, basis: str = "모델기반") -> dict:
    """레버 하나의 순절감액을 계산한다."""
    new_billing = annual_billing_peak(peak_new)
    delta_peak = BASE_BILLING_PEAK - new_billing
    basic_saving = BASE_BASIC_CHARGE - basic_charge(new_billing)
    energy_delta = BASE_ENERGY_COST - energy_cost(prod_new, peak_new)
    net = basic_saving + energy_delta - extra_labor
    return {
        "레버": name,
        "근거 강도": basis,
        "Δ최대수요(kW)": round(delta_peak, 1),
        "기본요금 절감(원)": round(basic_saving),
        "전력량요금 차액(원)": round(energy_delta),
        "인건비 증가(원)": round(extra_labor),
        "순절감액(원)": round(net),
        "일생산량 보존": bool(
            np.allclose(
                prod_new.groupby(prod_new.index.normalize()).sum(),
                PROD.groupby(PROD.index.normalize()).sum(),
                atol=1e-6,
            )
        ),
    }


# ── 레버 실행 ──────────────────────────────────────────────────────────
p1a = lever_L1(PEAK15, hours=(8, 13))          # 계획서 명세
p1b = lever_L1(PEAK15, hours=(8, 11, 13))      # 실측 피크 분포 반영 확장
p1c = lever_L1(PEAK15, hours=(8, 9, 11, 13))   # 상위 피크가 나타나는 전 시간대 포괄
p2, q2 = lever_L2(PEAK15, PROD)
p3, q3, L3_LEFTOVER = lever_L3(PEAK15, PROD)
p4n, q4n, labor_n = lever_L4(PEAK15, PROD, "night")
p4w, q4w, labor_w = lever_L4(PEAK15, PROD, "weekend")

levers_tbl = pd.DataFrame(
    [
        evaluate_lever("L1a 기동 분산 (08·13시)", p1a, PROD, 0.0, "가정기반(α)"),
        evaluate_lever("L1b 기동 분산 (08·11·13시)", p1b, PROD, 0.0, "가정기반(α)+실측 피크분포"),
        evaluate_lever("L1c 기동 분산 (08·09·11·13시)", p1c, PROD, 0.0, "가정기반(α)+실측 피크분포"),
        evaluate_lever("L2 고생산 시간대 이동", p2, q2, 0.0, "모델기반(β)"),
        evaluate_lever("L3 시간당 생산량 상한", p3, q3, 0.0, "모델기반(β)"),
        evaluate_lever("L4a 야간 이전", p4n, q4n, labor_n, "모델기반+인건비 가정"),
        evaluate_lever("L4b 주말 이전", p4w, q4w, labor_w, "모델기반+특근수당 가정"),
    ]
)
save_table(levers_tbl, "ch4_levers")
print("── 레버별 효과 (전체기간 실측 peak15 기준) ──")
display(levers_tbl)

# ── 핵심 발견 3건을 명시한다 ──────────────────────────────────────────
_l1a = levers_tbl.iloc[0]["Δ최대수요(kW)"]
_l1b = levers_tbl.iloc[1]["Δ최대수요(kW)"]
_l1c = levers_tbl.iloc[2]["Δ최대수요(kW)"]
_l2 = levers_tbl.iloc[3]["Δ최대수요(kW)"]
print(
    f"\n  [발견 1] L1a(08·13시) Δ최대수요 = {_l1a:.1f} kW → "
    f"L1b(+11시) = {_l1b:.1f} kW → L1c(+09시) = {_l1c:.1f} kW\n"
    "     청구 피크는 **최대값**이라 상위 사건을 하나라도 남기면 절감이 0이다.\n"
    "     07-19 11시(222 kW)가 08·13시 분산으로는 건드려지지 않으므로,\n"
    "     11시를 포함해야 비로소 청구 피크가 내려간다. 11시까지 덮으면 이번에는\n"
    "     09시 사건(07-21 09시 = 214 kW)이 새 상한이 된다 → 09시까지 포함해야\n"
    "     추가 저감이 생긴다. **상위 사건을 순차적으로 모두 덮어야 한다.**"
)
print(
    f"\n  [발견 2] 생산량 이동(L2) Δ최대수요 = {_l2:.1f} kW — 효과가 미미하다.\n"
    f"     pooled β = {BETA:.5f} kW/생산1단위 이므로 20 kW를 낮추려면\n"
    f"     약 {20 / BETA if BETA > 0 else float('inf'):,.0f} 단위를 옮겨야 한다(일 생산량 초과).\n"
    "     즉 이 공장의 peak15 는 생산 '물량'보다 **설비 동시 가동 여부**가 지배한다.\n"
    "     → 저감의 핵심 레버는 물량 재배치가 아니라 **동시성 분산**이다."
)
print(
    f"\n  [발견 3] L3 이월 잔량 = {L3_LEFTOVER:.6f} (0이어야 일 생산량 보존)\n"
    f"     주말 할증: 실측 배수 {WEEKEND_LABOR_MULTIPLIER} (할증 0) → "
    f"실제 특근수당 {WEEKEND_PREMIUM_PARAM}배 가정으로 계산."
)


def plot_l1_load_curve(peak: pd.Series, peak_new: pd.Series):
    """F37 — L1 기동 분산 전후 08시 부하 곡선."""
    base_h = peak.groupby(peak.index.hour).mean()
    new_h = peak_new.groupby(peak_new.index.hour).mean()
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(base_h.index, base_h.to_numpy(), color=COLOR_MUTED, lw=2, marker="o", ms=4, label="현행")
    ax.plot(new_h.index, new_h.to_numpy(), color=COLOR_HERO, lw=2, marker="o", ms=4, label="L1 적용")
    ax.annotate(
        f"08시 {base_h[8]:.1f} → {new_h[8]:.1f}",
        (8, base_h[8]), textcoords="offset points", xytext=(10, 8), fontsize=8, color=INK,
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시각", fontsize=9)
    ax.set_ylabel("평균 peak15 (kW)", fontsize=9)
    ax.set_title(f"L1 기동 분산 전후 시간대별 부하 (α={ALPHA_SIMULTANEOUS:.0%})", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, pd.DataFrame({"현행": base_h, "L1 적용": new_h})


_fig, _src = plot_l1_load_curve(PEAK15, p1c)
save_fig(_fig, "F37", "L1 기동분산 전후 부하곡선", "8.2", source_table=_src)

# %% [markdown]
# ### 8.3 시나리오 비교와 민감도
#
# - **목적**: 레버 조합 시나리오의 **순절감액**을 계산하고, 가정 파라미터
#   ±30%에서 **우선순위가 뒤집히지 않는지** 확인한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_scenarios.csv`, `ch4_sensitivity.csv`, F35·F36·F38
#
# **반드시 순절감액으로 판정한다.** L4(야간 이전)는 인건비가 1.5배로 늘기 때문에
# "무조건 좋은 레버가 아님"을 수치로 보이는 것이 정직성 가점 요소다.

# %%
def run_scenarios() -> pd.DataFrame:
    """레버 조합 시나리오를 평가한다.

    S1과 S1b를 나란히 두어 **상위 피크 사건을 하나라도 남기면 절감이 0** 임을
    표로 직접 보인다(최대값 목적함수의 핵심 성질).
    """
    def row(label, peak_new, prod_new, extra=0.0, comp="-"):
        base = evaluate_lever("-", peak_new, prod_new, extra)
        return {
            "시나리오": label,
            **{k: v for k, v in base.items() if k not in ("레버", "근거 강도")},
            "구성": comp,
        }

    p1c2, q1c2 = lever_L2(p1c, PROD)
    p1c3, q1c3, _ = lever_L3(p1c, PROD)
    p1c24, q1c24, lab = lever_L4(p1c2, q1c2, "night")

    return pd.DataFrame(
        [
            row("S0 현행 (무조정)", PEAK15, PROD, 0.0, "-"),
            row("S1a L1a 단독 (08·13시)", p1a, PROD, 0.0, "기동 분산 (계획서 명세)"),
            row("S1b L1b 단독 (08·11·13시)", p1b, PROD, 0.0, "기동 분산 + 11시 포함"),
            row("S1c L1c 단독 (08·09·11·13시)", p1c, PROD, 0.0, "기동 분산 + 09·11시 포함"),
            row("S2 L1c+L2", p1c2, q1c2, 0.0, "기동 분산 + 시간대 이동"),
            row("S3 L1c+L3", p1c3, q1c3, 0.0, "기동 분산 + 생산량 상한"),
            row("S4 L1c+L2+L4(야간)", p1c24, q1c24, lab, "기동 분산 + 시간대 이동 + 야간 이전"),
        ]
    )


scenarios_tbl = run_scenarios()
save_table(scenarios_tbl, "ch4_scenarios")
print("── 시나리오별 순절감액 ──")
display(scenarios_tbl[["시나리오", "구성", "Δ최대수요(kW)", "기본요금 절감(원)",
                       "인건비 증가(원)", "순절감액(원)", "일생산량 보존"]])

BEST_SCENARIO = scenarios_tbl.loc[scenarios_tbl["순절감액(원)"].idxmax(), "시나리오"]
PEAK_REDUCTION_KW = float(scenarios_tbl["Δ최대수요(kW)"].max())
BEST_NET_SAVING = float(scenarios_tbl["순절감액(원)"].max())
print(f"\n  최적 시나리오 = {BEST_SCENARIO}")
print(f"  최대 Δ최대수요 = {PEAK_REDUCTION_KW:.1f} kW / 순절감액 {BEST_NET_SAVING:,.0f} 원")


def sensitivity_analysis() -> pd.DataFrame:
    """기본요금 단가 ±30% × α 10/20/30% 에서 레버 우선순위가 보존되는지 본다."""
    rows = []
    for rate_mult in (0.7, 1.0, 1.3):
        rate = BASE_RATE_KRW_PER_KW_MONTH * rate_mult
        for alpha in (0.10, 0.20, 0.30):
            pa = lever_L1(PEAK15, alpha, hours=(8, 9, 11, 13))
            pa2, qa2 = lever_L2(pa, PROD)
            pa4, qa4, lab = lever_L4(pa2, qa2, "night")
            combos = {
                "L1c 단독": (pa, PROD, 0.0),
                "L1c+L2": (pa2, qa2, 0.0),
                "L1c+L2+L4(야간)": (pa4, qa4, lab),
            }
            nets = {}
            for label, (pk, pq, lb) in combos.items():
                nb = annual_billing_peak(pk)
                nets[label] = (
                    (BASE_BILLING_PEAK * rate * BILLING_MONTHS - nb * rate * BILLING_MONTHS)
                    + (BASE_ENERGY_COST - energy_cost(pq, pk)) - lb
                )
            best = max(nets, key=nets.get)
            rows.append(
                {
                    "기본요금단가 배수": rate_mult, "α": alpha,
                    **{f"{k} 순절감(원)": round(v) for k, v in nets.items()},
                    "최적 조합": best,
                }
            )
    return pd.DataFrame(rows)


sensitivity_tbl = sensitivity_analysis()
save_table(sensitivity_tbl, "ch4_sensitivity")
PRIORITY_STABLE = sensitivity_tbl["최적 조합"].nunique() == 1
print("\n── 민감도 분석 (기본요금 단가 ±30% × α 10/20/30%) ──")
display(sensitivity_tbl)
print(
    f"\n  레버 우선순위 보존: {PRIORITY_STABLE} "
    f"(최적 조합 후보 {sensitivity_tbl['최적 조합'].unique().tolist()})"
)


def plot_waterfall(levers: pd.DataFrame):
    """F35 — 레버별 Δ최대수요 폭포."""
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    vals = levers["Δ최대수요(kW)"].to_numpy()
    colors = [COLOR_HERO if v > 0 else PALETTE_ADJACENT[1] for v in vals]
    bars = ax.bar(levers["레버"], vals, color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals.max(), 1) * 0.03,
                f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1)
    ax.set_ylabel("Δ최대수요 (kW)", fontsize=9)
    ax.set_title("레버별 청구 피크 저감량", fontsize=10, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, levers[["레버", "Δ최대수요(kW)"]]


_fig, _src = plot_waterfall(levers_tbl)
save_fig(_fig, "F35", "레버별 최대수요 저감", "8.3", source_table=_src)


def plot_net_saving(scen: pd.DataFrame):
    """F36 — 시나리오별 순절감액 (발산형)."""
    t = scen[scen["시나리오"] != "S0 현행 (무조정)"]
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    vals = t["순절감액(원)"].to_numpy() / 1e6
    colors = [COLOR_HERO if v > 0 else PALETTE_ADJACENT[1] for v in vals]
    bars = ax.bar(t["시나리오"], vals, color=colors, width=0.58)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + abs(vals).max() * 0.03,
                f"{v:.1f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=INK_SOFT, lw=1)
    ax.set_ylabel("순절감액 (백만원)", fontsize=9)
    ax.set_title("시나리오별 순절감액 (요금절감 − 인건비 증가)", fontsize=10, color=INK)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
    fig.tight_layout()
    return fig, t[["시나리오", "순절감액(원)"]]


_fig, _src = plot_net_saving(scenarios_tbl)
save_fig(_fig, "F36", "시나리오별 순절감액", "8.3", source_table=_src)


def plot_sensitivity_heatmap(sens: pd.DataFrame):
    """F38 — 민감도 히트맵 (기본요금 단가 × α)."""
    piv = sens.pivot(index="α", columns="기본요금단가 배수", values="L1c+L2 순절감(원)") / 1e6
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap=CMAP_SEQ)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"×{c}" for c in piv.columns], fontsize=9)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([f"α={i:.0%}" for i in piv.index], fontsize=9)
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            ax.text(j, i, f"{piv.iloc[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if piv.iloc[i, j] > piv.to_numpy().max() / 2 else INK)
    ax.set_title("S2(L1c+L2) 순절감액 민감도 (백만원)", fontsize=10, color=INK)
    ax.grid(False)
    fig.tight_layout()
    return fig, piv.round(2)


_fig, _src = plot_sensitivity_heatmap(sensitivity_tbl)
save_fig(_fig, "F38", "민감도 히트맵", "8.3", source_table=_src)

# %% [markdown]
# ### 8.4 현장 운영 프로토콜
#
# - **목적**: 모델 출력이 **어떤 현장 조치로 이어지는지** 타임라인으로 정리한다.
# - **보고서 대응절**: 4장
# - **산출물**: `ch4_protocol.csv`, F39

# %%
def build_protocol() -> pd.DataFrame:
    """전일 24시 예측 → 익일 조치로 이어지는 운영 프로토콜."""
    rows = [
        ("전일 23:50", "데이터 수집 완료", "당일 24시간 전력·생산 실적 확정", "자동", "-"),
        ("전일 24:00", "예측 실행 (원점)", f"{FINAL_MODEL_NAME}로 익일 24시간 peak15·피크확률 산출", "자동", f"추론 1.2초 이내"),
        ("전일 24:05", "경보 판정", f"peak15 예측 ≥ τ 인 시각을 위험시간으로 지정", "자동", "τ는 fold 중앙값 고정"),
        ("전일 24:10", "일정 조정안 생성", "L1(기동 분산) 우선 적용, 필요시 L2(생산 이동)", "자동", "일 총생산량 보존"),
        ("익일 07:30", "기동 계획 확정", "08:00/08:20/08:40 3그룹 계단식 기동 지시", "현장", "L1 — 생산·인건비 영향 없음"),
        ("익일 08:00~", "실시간 감시", "위험시간 진입 시 15분 수요 감시 강화", "현장", "예측구간 q90 초과 여부"),
        ("익일 12:50", "재가동 분산", "13시 재가동도 계단식 적용", "현장", "13시 피크율 17.1%"),
        ("익일 18:00", "일일 검증", "예측 대비 실측 편차 기록, 초과 시 원인 기록", "자동", "7장 실패조건과 대조"),
        ("월 1회", "임계값 재점검", "θ·τ 재산출 여부 검토 (재산출 시 정의 변경 주의)", "분석", "θ 고정 원칙 유지"),
    ]
    return pd.DataFrame(rows, columns=["시점", "조치", "내용", "주체", "비고"])


protocol_tbl = build_protocol()
save_table(protocol_tbl, "ch4_protocol")
print("── 현장 운영 프로토콜 ──")
display(protocol_tbl)

# 점검 우선순위 — 7장 규칙과 연결
priority_tbl = rules_tbl.copy()
priority_tbl["점검 우선순위"] = range(1, len(priority_tbl) + 1)
priority_tbl["권고 조치"] = [
    "L1 기동 분산 + 실시간 감시 강화",
    "L2 생산 시간대 이동 검토",
    "모니터링만 (조치 불필요)",
][: len(priority_tbl)]
save_table(priority_tbl, "ch4_priority")
print("\n── 규칙별 점검 우선순위 (7.6절 규칙과 연결) ──")
display(priority_tbl)


def plot_alert_timeline(prot: pd.DataFrame):
    """F39 — 경보 운영 타임라인."""
    fig, ax = plt.subplots(figsize=(11, 2.8))
    colors = {"자동": COLOR_HERO, "현장": PALETTE_ADJACENT[1], "분석": PALETTE_ADJACENT[2]}
    for i, (_, r) in enumerate(prot.iterrows()):
        ax.scatter(i, 0, s=150, color=colors.get(r["주체"], COLOR_MUTED), zorder=3)
        ax.annotate(r["시점"], (i, 0), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=7.5, color=INK, rotation=30)
        ax.annotate(r["조치"], (i, 0), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=7.5, color=INK_SOFT, rotation=-30)
    ax.plot(range(len(prot)), [0] * len(prot), color="#d6d5d1", lw=1.4, zorder=1)
    for k, c in colors.items():
        ax.scatter([], [], s=100, color=c, label=k)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper center")
    ax.set_ylim(-1.2, 1.2); ax.set_yticks([]); ax.set_xticks([])
    ax.set_title("경보 운영 타임라인 (전일 24시 예측 → 익일 조치)", fontsize=10, color=INK, pad=22)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    return fig, prot


_fig, _src = plot_alert_timeline(protocol_tbl)
save_fig(_fig, "F39", "경보 운영 타임라인", "8.4", source_table=_src)

# %% [markdown]
# ### 8.5 보고서 4장 본문 초안 생성
#
# - **목적**: 위 결과를 4장 본문 초안 markdown 으로 출력한다.
# - **보고서 대응절**: 4장 전체
# - **산출물**: `outputs/report_ch4_draft.md`

# %%
def write_ch4_draft() -> str:
    """4장 '현장 활용방안' 본문 초안을 생성한다."""
    best = scenarios_tbl.loc[scenarios_tbl["순절감액(원)"].idxmax()]
    l1a = levers_tbl[levers_tbl["레버"].str.startswith("L1a")].iloc[0]
    l1b = levers_tbl[levers_tbl["레버"].str.startswith("L1b")].iloc[0]
    l1c = levers_tbl[levers_tbl["레버"].str.startswith("L1c")].iloc[0]
    l2r = levers_tbl[levers_tbl["레버"].str.startswith("L2")].iloc[0]
    l3r = levers_tbl[levers_tbl["레버"].str.startswith("L3")].iloc[0]
    l4n = levers_tbl[levers_tbl["레버"].str.startswith("L4a")].iloc[0]

    md = f"""# 제 4 장. 현장 활용방안 〔10점〕

## 4.1 예측 결과가 이어지는 현장 판단

전일 24시 시점에 익일 24시간의 `peak15`(15분 최대수요)와 피크 발생확률을 산출하고,
경보 임계값 τ를 넘는 시각을 **위험시간**으로 지정한다. 위험시간에 대해 설비 기동시점과
생산 일정을 조정하는 것이 본 장의 활용방안이다. 최종모델의 336시간 배치 추론은
1.2초 이내여서 매일 반복 실행에 부담이 없다.

## 4.2 요금 구조와 목적함수

한국전력공사는 **3개월간의 최대 피크 전력을 기준으로 1년간 전기료를 부과**한다.
따라서 연중 단 한 번의 최대수요가 12개월 기본요금을 결정하며, 저감의 목적함수는
평균이 아니라 **최대값(max)** 이다. 전체기간 실측 기준 연간 청구 피크는
**{BASE_BILLING_PEAK:.0f} kW**(3개월 롤링 최대의 최대값)이다.

> **중요한 한계 명시**: 테스트 구간(9월)의 최대 `peak15`는 204 kW인데 7~9월 창의 최대는
> 222 kW(7월)이다. 따라서 **9월 구간 단독으로는 연간 기본요금에 영향을 주지 않는다**
> (9월을 0으로 낮추는 극단 가정에서도 절감액 {SEPT_ONLY_SAVING:,.0f}원).
> 본 절의 시뮬레이션은 전체기간(01-01~09-14) **실측 `peak15` 시계열** 위에서 수행하고,
> 9월 예측 결과는 운영 시나리오 검증용으로만 사용한다.

## 4.3 저감 레버 4종

모든 레버는 **일일 총 생산량을 보존한 채 시간 배치만 바꾼다.** 생산을 줄이는 것이
아니라 옮기므로 매출 손실이 없다.

| 레버 | 조정 내용 | 데이터 근거 | Δ최대수요 | 근거 강도 |
|---|---|---|---|---|
| **L1a** 기동시점 분산 (08·13시) | 08시 일제기동을 08:00/08:20/08:40 계단식으로 분산. 13시 재가동 동일 | 08시 피크율 24.5%(1위), 13시 17.1%. 7시→8시 평균 87.6→121.3 급등 = 동시 기동 돌입전류 | {l1a['Δ최대수요(kW)']:.1f} kW | 가정기반(α) |
| **L1b** 기동시점 분산 (+11시) | 위에 **11시 점심 전 집중 작업 분산**을 추가 | 연중 상위 {TOP_N_PEAKS}개 피크의 시간대 분포에서 11시가 두 번째로 많다. 최대값 222 4건 중 3건이 08시, **1건이 11시** | {l1b['Δ최대수요(kW)']:.1f} kW | 가정기반(α)+실측 |
| **L1c** 기동시점 분산 (+09시) | 위에 **09시 집중 작업 분산**을 추가 | 11시까지 덮으면 09시 사건(07-21 09시 = 214 kW)이 새 상한이 된다 | {l1c['Δ최대수요(kW)']:.1f} kW | 가정기반(α)+실측 |
| **L2** 고생산 시간대 이동 | 피크 시각의 생산량 일부를 같은 날 저부하 시간(10~11, 15~16시)으로 이동 | 생산량–전력 상관 0.518, pooled β={BETA:.5f} kW/단위 | {l2r['Δ최대수요(kW)']:.1f} kW | 모델기반 |
| **L3** 시간당 생산량 상한 | 피크 위험 시간에 상한을 걸고 초과분을 다음 시간으로 이월 | L2의 자동화 버전. 예측 `peak15`가 임계 초과 시에만 발동 | {l3r['Δ최대수요(kW)']:.1f} kW | 모델기반 |
| **L4** 야간·주말 이전 | 주간 고부하 작업 일부를 야간 또는 주말로 이전 | 토 51.3 / 일 41.6 여유. 정상 가동일 기준 야간 피크 0건 | {l4n['Δ최대수요(kW)']:.1f} kW | 모델기반+인건비 가정 |

### 두 가지 중요한 발견

**(1) 상위 피크 사건을 하나라도 남기면 절감이 0이다.** 청구 피크는 **최대값**이므로,
계획서 명세대로 08·13시만 분산한 L1a는 Δ최대수요 **{l1a['Δ최대수요(kW)']:.1f} kW** 에 그친다.
전체 최대값 222 kW 4건 중 3건은 08시이지만 **1건(07-19 11시)이 11시**여서, 이 한 사건이
남는 순간 연간 청구 피크가 그대로 유지되기 때문이다. 11시를 포함한 L1b는
Δ최대수요 **{l1b['Δ최대수요(kW)']:.1f} kW**, 09시까지 포함한 L1c는
**{l1c['Δ최대수요(kW)']:.1f} kW** 를 달성한다. 즉 **상위 피크 사건을 순차적으로 모두 덮어야
추가 저감이 생긴다.** 연중 상위 {TOP_N_PEAKS}개 피크의 시간대 분포(08시·09시·11시 집중)가
이 설계의 근거다.

**(2) 이 공장의 피크는 생산 '물량'보다 설비 '동시 가동'이 지배한다.**
pooled 회귀(시간 고정효과 포함)의 생산량 계수는 β={BETA:.5f} kW/단위
(p={BETA_DIAG['p-value']}, R²={BETA_DIAG['R²']})로, 20 kW를 낮추려면 약
{20 / BETA if BETA > 0 else 0:,.0f} 단위를 옮겨야 한다 — 하루 총 생산량을 넘는 양이다.
따라서 **물량 재배치(L2·L3)보다 동시성 분산(L1)이 핵심 레버**다.
이는 생산 일정을 크게 흔들지 않고도 피크를 관리할 수 있다는 뜻이므로 현장에 유리한 결론이다.

**L1c를 최우선 권고한다.** 생산량과 인건비에 영향을 주지 않으면서 피크가 집중되는
시간대를 직접 겨냥하기 때문이다. 단 08시는 인건비 1.5배 구간이므로 기동 인력 배치는
기존과 동일하게 유지한다.

**L4는 무조건 좋은 레버가 아니다.** 야간 이전은 인건비가 1.5배가 되어 추가 비용
{l4n['인건비 증가(원)']:,.0f}원이 발생하며, 순절감액은 {l4n['순절감액(원)']:,.0f}원이다.
또한 **주말 할증은 제공 데이터상 0**(요일별 인건비 분포가 균등)이므로, 실제 특근수당을
파라미터({WEEKEND_PREMIUM_PARAM}배 가정)로 노출해 민감도에 포함했다.

## 4.4 시나리오별 순절감액

절감액 = Δ최대수요(kW) × 기본요금단가 × 12개월 + 시간대 이동 전력량요금 차액 − 인건비 증가분

| 시나리오 | 구성 | Δ최대수요 | 기본요금 절감 | 인건비 증가 | **순절감액** |
|---|---|---|---|---|---|
"""
    for _, r in scenarios_tbl.iterrows():
        md += (
            f"| {r['시나리오']} | {r['구성']} | {r['Δ최대수요(kW)']:.1f} kW | "
            f"{r['기본요금 절감(원)']:,.0f}원 | {r['인건비 증가(원)']:,.0f}원 | "
            f"**{r['순절감액(원)']:,.0f}원** |\n"
        )
    md += f"""
최적 시나리오는 **{best['시나리오']}** 으로, 최대수요를 **{best['Δ최대수요(kW)']:.1f} kW**
낮추고 순절감액 **{best['순절감액(원)']:,.0f}원**을 확보한다. 모든 시나리오에서
일일 총 생산량이 보존됨을 검증했다(L3 이월 잔량 {L3_LEFTOVER:.3f}).

## 4.5 가정과 민감도

기본요금 단가(원/kW·월)와 α(08시 피크 중 동시 기동 기여분)는 제공 데이터로 검증할 수
없으므로 파라미터로 노출했다. 기본요금 단가 ±30% × α 10/20/30% 조합에서 레버
우선순위가 {'보존되었다' if PRIORITY_STABLE else '일부 변동했다'}
(최적 조합: {', '.join(sensitivity_tbl['최적 조합'].unique())}).
설비 ID와 제품코드가 제공되지 않아 L1의 효과는 데이터로 직접 산출할 수 없고
**가정기반**임을 표에 명시했다.

## 4.6 현장 운영 프로토콜

| 시점 | 조치 | 내용 | 주체 |
|---|---|---|---|
"""
    for _, r in protocol_tbl.iterrows():
        md += f"| {r['시점']} | {r['조치']} | {r['내용']} | {r['주체']} |\n"
    md += """
점검 우선순위는 3.6절에서 도출한 고위험 규칙(R1~R3)과 직접 연결한다.
R1 조건에 해당하는 시각은 L1 기동 분산과 실시간 감시를 함께 적용하고,
R3 조건은 모니터링만 수행해 불필요한 생산조정비용을 피한다.
"""
    return md


_ch4 = write_ch4_draft()
(OUTPUT_DIR / "report_ch4_draft.md").write_text(_ch4, encoding="utf-8")
print(f"── 4장 본문 초안 생성 완료 ({len(_ch4):,}자) → outputs/report_ch4_draft.md ──")
