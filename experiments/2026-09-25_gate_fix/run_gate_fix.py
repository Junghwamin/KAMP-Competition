"""1단계 분류기 휴무 오분류 수정안 비교 — 계획은 PLAN.md.

순서: ① OOF 로 수정안 판정(테스트 미사용) → ② 판정 뒤 테스트 영향 보고 → ③ 평일 계획 휴무 가정 시나리오.

실행 (저장소 루트에서):
    PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiments/2026-09-25_gate_fix/run_gate_fix.py
"""
from __future__ import annotations

import ast
import contextlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                  # experiments/<실험>/ → 저장소 루트
# 재실행 검증 때는 KAMP_EXP_OUT·KAMP_EXP_FIG 로 결과를 다른 곳에 쓴다(보관된 결과를 덮지 않게)
OUT = Path(os.environ.get("KAMP_EXP_OUT") or HERE / "results").resolve()   # chdir 전에 절대경로로 고정
FIG = Path(os.environ.get("KAMP_EXP_FIG") or HERE / "figures").resolve()
# 파이프라인(s00~s03) import 는 작업 디렉터리의 outputs/ 에 표·그림을 쓴다 → 저장소가 아닌 _work/ 에서 실행한다
WORK = HERE / "_work"

if os.environ.get("PYTHONHASHSEED") != "42" or os.environ.get("KAMP_FAST", "0") != "0":
    sys.exit("PYTHONHASHSEED=42, KAMP_FAST=0 으로 실행해야 한다")
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
(WORK / "data").mkdir(parents=True, exist_ok=True)
if not (WORK / "data" / "okm_augumented_2021.csv").exists():
    shutil.copy2(REPO / "data" / "okm_augumented_2021.csv", WORK / "data" / "okm_augumented_2021.csv")
os.chdir(WORK)
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    import s00_env as s00  # noqa: E402
    import s01_diagnose as s01  # noqa: E402
    import s02_features as s02  # noqa: E402
    import s03_split as s03  # noqa: E402
import lightgbm as lgb  # noqa: E402

assert s00.FAST is False
print(f"[준비] s00~s03 {time.perf_counter() - T0:.1f}s", flush=True)

# s05 원문에서 레짐 모델·CV 루프만 추출 (기준 A 재현용)
WANT = {"LGB_BASE", "N_TRIALS", "N_ESTIMATORS", "_TIMING", "run_cv", "REGIME_CUT", "make_regime_model"}
ns: dict = {}
for _m in (s00, s01, s02, s03):
    ns.update(vars(_m))
ns.update(time=time, lgb=lgb)
_nodes = []
for node in ast.parse((REPO / "src" / "s05_models.py").read_text(encoding="utf-8")).body:
    names = set()
    if isinstance(node, ast.FunctionDef):
        names = {node.name}
    elif isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = {node.target.id}
    if names & WANT:
        _nodes.append(node)
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "s05_models.py", "exec"), ns)
assert ns["N_ESTIMATORS"] == 800
run_cv, make_regime_model = ns["run_cv"], ns["make_regime_model"]
LGB_BASE, N_EST, REGIME_CUT = ns["LGB_BASE"], ns["N_ESTIMATORS"], ns["REGIME_CUT"]

THETA = s01.THETA
CAL = s01.operating_calendar
FG = s02.FEATURE_GROUPS
FEATURE_COLS = list(s02.FEATURE_COLS)
ACTIVE_FOLDS = s03.ACTIVE_FOLDS
NOISE_MAE, NOISE_PEAK = 0.200, 0.962          # 제출 노트북 6.6절 잡음 폭 (PLAN §3)
TOP_DAYS = pd.to_datetime(["2021-08-02", "2021-08-03"])
GATE_22 = FG["시간정보"] + FG["공정상태"] + FG["생산정보"]
assert len(GATE_22) == 22


# ── 수정안 모델 — s05 make_regime_model 과 같은 코드에 게이트 부분만 바꾼다 ──────────
def make_regime_model_v(gate_features=None, shutdown_override: bool = False):
    """3분류 레짐 모델. gate_features=None·shutdown_override=False 면 현행과 같아야 한다(검증함)."""

    def regime_label(y_avg):
        return np.digitize(np.asarray(y_avg), [30.0, REGIME_CUT])

    def fit_fn(Xtr, ytr, Xva):
        gcols = [c for c in (gate_features or Xtr.columns) if c in Xtr.columns]
        rtr = regime_label(ytr["y_avg"])
        gate = lgb.LGBMClassifier(n_estimators=400, **{**LGB_BASE, "objective": "multiclass"})
        gate.fit(Xtr[gcols], rtr)
        regs: dict = {}
        fallbacks: dict = {}
        for tgt in ("y_avg", "y_peak"):
            regs[tgt] = {}
            for r in np.unique(rtr):
                m_tr = rtr == r
                if m_tr.sum() < 20:
                    continue
                reg = lgb.LGBMRegressor(n_estimators=N_EST, **LGB_BASE)
                reg.fit(Xtr[m_tr], ytr[tgt][m_tr])
                regs[tgt][int(r)] = reg
            fb = lgb.LGBMRegressor(n_estimators=N_EST, **LGB_BASE)
            fb.fit(Xtr, ytr[tgt])
            fallbacks[tgt] = fb

        def route(X) -> np.ndarray:
            rr = gate.predict(X[gcols])
            if shutdown_override:
                # 계획상 휴무일(일생산량 0)은 게이트 출력과 관계없이 기저 레짐으로 보낸다
                assert 0 in regs["y_avg"], "기저 레짐 회귀모델이 없다"
                rr = np.where(X["is_shutdown"].to_numpy() == 1, 0, rr)
            return rr

        def predict(X, tgt: str) -> np.ndarray:
            rr = route(X)
            out = np.empty(len(X), float)
            assigned = np.zeros(len(X), bool)
            for r, reg in regs[tgt].items():
                m = rr == r
                if m.any():
                    out[m] = reg.predict(X[m])
                    assigned |= m
            if (~assigned).any():
                out[~assigned] = fallbacks[tgt].predict(X[~assigned])
            return out

        return {
            "pred_avg": predict(Xva, "y_avg"),
            "pred_peak": predict(Xva, "y_peak"),
            "_route": route,
            "_predict": predict,
            "_gate_splits": pd.Series(gate.booster_.feature_importance("split"), index=gate.booster_.feature_name()),
        }

    return fit_fn


VARIANTS = {
    "A": ("현행", dict()),
    "R": ("계획 휴무 강제", dict(shutdown_override=True)),
    "G": ("게이트 입력 제한(22개)", dict(gate_features=GATE_22)),
    "RG": ("R + G", dict(gate_features=GATE_22, shutdown_override=True)),
}


def day_flags(index) -> dict:
    day = index.normalize()
    st = CAL["is_shutdown"].astype(int)
    s0 = st.reindex(day).to_numpy()
    s7 = st.reindex(day - pd.Timedelta(days=7)).to_numpy()
    return {"정상일": s0 == s7, "토요일": index.dayofweek == 5, "휴무일": s0 == 1}


def summarize(res: dict, base: dict | None, vid: str) -> dict:
    oof = res["oof"]
    y, p = oof["y_avg"].to_numpy(), oof["pred_avg"].to_numpy()
    reg = s03.regression_metrics(oof["y_avg"], oof["pred_avg"], THETA)
    cls = s03.classification_metrics(oof["y_cls"], (oof["pred_peak"] >= res["tau"]).astype(int), oof["pred_peak"])
    fm = res["fold_metrics"].set_index("fold")["MAE"]
    ae = np.abs(y - p)
    fl = day_flags(oof.index)
    keep = ~oof.index.normalize().isin(TOP_DAYS)
    row = {
        "ID": vid, "수정안": VARIANTS[vid][0], "조건": res["cond"],
        "OOF MAE": reg["MAE"], "Peak-MAE": reg["Peak-MAE"], "Recall": cls["Recall"], "F1": cls["F1"],
        "FP": cls["FP"], "FN": cls["FN"], "τ": res["tau"], "fold 표준편차": float(fm.std()),
        **{f"fold{f}": float(fm[f]) for f in ACTIVE_FOLDS},
        "정상일 MAE": float(ae[fl["정상일"]].mean()),
        "휴무일 MAE": float(ae[fl["휴무일"]].mean()),
        "이틀 제외 MAE": float(ae[keep].mean()),
        "이틀 MAE": float(ae[~keep].mean()),
    }
    if base is None:
        return row
    assert base["oof"].index.equals(oof.index)
    b = summarize(base, None, "A")
    for k in ("OOF MAE", "Peak-MAE", "Recall", "F1", "fold 표준편차", *[f"fold{f}" for f in ACTIVE_FOLDS],
              "정상일 MAE", "휴무일 MAE", "이틀 제외 MAE", "이틀 MAE"):
        row[f"Δ{k}"] = row[k] - b[k]
    # paired_bootstrap_mae 의 diff = MAE(기준) − MAE(변형) → 부호를 뒤집어 '변형 − 기준'
    bs = s03.paired_bootstrap_mae(y, base["oof"]["pred_avg"].to_numpy(), p, index=oof.index)
    row["ΔMAE CI 하한"], row["ΔMAE CI 상한"], row["p"] = -bs["ci_high"], -bs["ci_low"], bs["p_value"]
    return row


def run_variant(vid: str, cond: str = "D1") -> dict:
    t = time.perf_counter()
    res = run_cv(f"gatefix-{vid}", make_regime_model_v(**VARIANTS[vid][1]), cond)
    print(f"  [{cond}] {vid:<3s} {VARIANTS[vid][0]:<16s} OOF MAE {s03.mae(res['oof']['y_avg'], res['oof']['pred_avg']):.4f} "
          f"({time.perf_counter() - t:.0f}s)", flush=True)
    return res


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════
# ① OOF 판정
print("[1] 교차검증(D1)", flush=True)
ref = run_cv("gatefix-원본A", make_regime_model(3), "D1")   # s05 원문 그대로
d1 = {v: run_variant(v) for v in VARIANTS}
assert np.array_equal(ref["oof"]["pred_avg"].to_numpy(), d1["A"]["oof"]["pred_avg"].to_numpy()), "수정안 코드가 현행과 다르다"
assert abs(s03.mae(ref["oof"]["y_avg"], ref["oof"]["pred_avg"]) - 11.0067) < 1e-4, "커밋된 OOF MAE 를 재현하지 못했다"
_daily = (d1["A"]["oof"]["y_avg"] - d1["A"]["oof"]["pred_avg"]).abs().groupby(d1["A"]["oof"].index.normalize()).mean()
assert set(_daily.nlargest(2).index) == set(TOP_DAYS), "오차 상위 2일이 계획과 다르다"
# R 은 휴무일이 아닌 행을 하나도 바꾸지 않아야 한다
_chg = d1["R"]["oof"]["pred_avg"].to_numpy() != d1["A"]["oof"]["pred_avg"].to_numpy()
_shut = s02.feat.loc[d1["A"]["oof"].index, "is_shutdown"].to_numpy() == 1
assert not (_chg & ~_shut).any(), "R 이 휴무일이 아닌 행을 바꿨다"
print(f"  재현 확인 OK · R 이 바꾼 행 {int(_chg.sum())}개 (모두 계획 휴무일)", flush=True)

tbl = pd.DataFrame([summarize(d1[v], None if v == "A" else d1["A"], v) for v in VARIANTS])
c = tbl["ID"] != "A"
tbl.loc[c, "① CI가 0 미포함"] = tbl.loc[c, "ΔMAE CI 상한"] < 0
tbl.loc[c, "② 정상 fold 악화 ≤ 잡음"] = (tbl.loc[c, "Δfold2"] <= NOISE_MAE) & (tbl.loc[c, "Δfold5"] <= NOISE_MAE)
tbl.loc[c, "③ 정상일 악화 ≤ 잡음"] = tbl.loc[c, "Δ정상일 MAE"] <= NOISE_MAE
tbl.loc[c, "④ Recall·Peak-MAE"] = (tbl.loc[c, "ΔRecall"] >= -0.01) & (tbl.loc[c, "ΔPeak-MAE"] <= NOISE_PEAK)
tbl.loc[c, "⑤ 이틀 제외 악화 ≤ 잡음"] = tbl.loc[c, "Δ이틀 제외 MAE"] <= NOISE_MAE

print("[2] D2 (⑥)", flush=True)
d2 = {v: run_variant(v, "D2") for v in VARIANTS}
d2_tbl = pd.DataFrame([summarize(d2[v], None if v == "A" else d2["A"], v) for v in VARIANTS])
tbl = tbl.merge(d2_tbl[["ID", "OOF MAE"]].rename(columns={"OOF MAE": "D2 OOF MAE"}), on="ID")
_d2a = float(d2_tbl.loc[d2_tbl["ID"] == "A", "OOF MAE"].iloc[0])
tbl["ΔD2 OOF MAE"] = tbl["D2 OOF MAE"] - _d2a
tbl.loc[c, "⑥ D2 개선"] = tbl.loc[c, "ΔD2 OOF MAE"] < 0
CRIT = ["① CI가 0 미포함", "② 정상 fold 악화 ≤ 잡음", "③ 정상일 악화 ≤ 잡음", "④ Recall·Peak-MAE",
        "⑤ 이틀 제외 악화 ≤ 잡음", "⑥ D2 개선"]
tbl.loc[c, "모두 통과"] = tbl.loc[c, CRIT].astype(bool).all(axis=1)

# 선택 규칙 (PLAN §3): 통과한 것 중 R > G > RG, 단 복잡한 쪽이 이틀 제외 MAE 를 잡음 폭보다 더 줄이면 그쪽
passed = [v for v in ("R", "G", "RG") if bool(tbl.set_index("ID").loc[v, "모두 통과"])]
CHOSEN = None
if passed:
    CHOSEN = passed[0]
    for v in passed[1:]:
        gain = tbl.set_index("ID").loc[CHOSEN, "이틀 제외 MAE"] - tbl.set_index("ID").loc[v, "이틀 제외 MAE"]
        if gain > NOISE_MAE:
            CHOSEN = v
print(f"  기준 통과: {passed or '없음'} → 채택: {CHOSEN}", flush=True)
save(tbl, "gate_fix_oof")
save(d2_tbl, "gate_fix_d2")

# 문제의 이틀 시간별 (그림·표용)
_hr = d1["A"]["oof"].loc[d1["A"]["oof"].index.normalize().isin(TOP_DAYS), ["y_avg"]].copy()
for v in VARIANTS:
    _hr[f"pred[{v}]"] = d1[v]["oof"].loc[_hr.index, "pred_avg"].to_numpy()
_hr.index.name = "ts"
_hr.reset_index().to_csv(OUT / "gate_fix_two_days_hourly.csv", index=False, encoding="utf-8-sig")

# ══════════════════════════════════════════════════════════════════════
# ② 판정 뒤 — 테스트 영향 (보고만)
print("[3] 테스트 영향 (판정에 쓰지 않음)", flush=True)
Xtr, ytr, Xte, yte, idx_te = s03.get_test_data("D1")
full_fit: dict = {}
test_rows = []
for v in VARIANTS:
    out = make_regime_model_v(**VARIANTS[v][1])(Xtr, ytr, Xte)
    full_fit[v] = out
    tau = d1[v]["tau"]
    reg = s03.regression_metrics(yte["y_avg"], out["pred_avg"], THETA)
    cls = s03.classification_metrics(yte["y_cls"], (out["pred_peak"] >= tau).astype(int), out["pred_peak"])
    test_rows.append({"ID": v, "수정안": VARIANTS[v][0], "τ(자기 OOF)": tau, "테스트 MAE": reg["MAE"],
                      "테스트 Peak-MAE": reg["Peak-MAE"], "Recall": cls["Recall"], "Precision": cls["Precision"],
                      "F1": cls["F1"], "TP": cls["TP"], "FP": cls["FP"], "FN": cls["FN"],
                      "A와 다른 예측 행 수": int((out["pred_avg"] != full_fit["A"]["pred_avg"]).sum()) if v != "A" else 0})
test_tbl = pd.DataFrame(test_rows)
assert abs(float(test_tbl.loc[0, "테스트 MAE"]) - 5.2199) < 1e-3, "현행 테스트 MAE(5.220)를 재현하지 못했다"
save(test_tbl, "gate_fix_test")

# ══════════════════════════════════════════════════════════════════════
# ③ 평일 계획 휴무 가정 시나리오 (추석 대리) — 9/1 이전 학습 모델로 9월 평일을 '휴무'로 가정해 예측
print("[4] 평일 계획 휴무 가정 시나리오", flush=True)
weekdays = [d for d in pd.date_range("2021-09-01", "2021-09-14", freq="D") if d.dayofweek < 5]
scen_rows = []
for d in weekdays:
    dfm = s01.df.copy()
    on_day = dfm.index.normalize() == d
    dfm.loc[on_day, "생산량"] = 0
    dfm.loc[on_day, "공장인원"] = 0
    calm = s01.build_operating_calendar(dfm)
    with contextlib.redirect_stdout(io.StringIO()):
        featm = s02.build_features(dfm, calm, THETA)
    Xd = featm.loc[on_day, FEATURE_COLS]
    assert (Xd["is_shutdown"] == 1).all()
    row = {"날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금"[d.dayofweek],
           "실제 일평균(가동일)": float(s01.df.loc[on_day, "y_avg"].mean())}
    for v in VARIANTS:
        pa = full_fit[v]["_predict"](Xd, "y_avg")
        rr = full_fit[v]["_route"](Xd)
        work = (Xd.index.hour >= 8) & (Xd.index.hour <= 17)
        row[f"{v} 예측 일평균"] = float(pa.mean())
        row[f"{v} 08~17시 평균"] = float(pa[work].mean())
        row[f"{v} 08~17시 가동 배정"] = int((rr[work] == 2).sum())
    scen_rows.append(row)
scen = pd.DataFrame(scen_rows)
save(scen, "gate_fix_whatif")
REF_SHUTDOWN = float(s01.df.loc[s01.df.index.normalize().isin(TOP_DAYS), "y_avg"].mean())

# ══════════════════════════════════════════════════════════════════════
# 그림
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COL = {"A": "#c0392b", "R": "#1f77b4", "G": "#2ca02c", "RG": "#9467bd"}

fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(_hr))
ax.plot(x, _hr["y_avg"], color="black", lw=2.2, label="실측")
for v in VARIANTS:
    ax.plot(x, _hr[f"pred[{v}]"], color=COL[v], lw=1.4, ls="-" if v in ("A", "R") else "--",
            label=f"{v} {VARIANTS[v][0]}")
ax.set_xticks([0, 8, 17, 24, 32, 41, 47])
ax.set_xticklabels(["8/2 0시", "8시", "17시", "8/3 0시", "8시", "17시", "23시"])
ax.set_ylabel("평균전력 (kW)")
ax.set_title("하계휴가 중 평일(계획 생산 0) — 교차검증 예측", fontsize=11)
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
# 1차 당시 그림 — 최종 그림(make_figures.py)을 덮지 않게 이름을 분리한다
fig.savefig(FIG / "round1_fig1_two_days.png", dpi=150)

fig, ax = plt.subplots(figsize=(9, 3.4))
xs = np.arange(len(scen))
w = 0.2
for i, v in enumerate(VARIANTS):
    ax.bar(xs + (i - 1.5) * w, scen[f"{v} 예측 일평균"], width=w, color=COL[v], label=f"{v} {VARIANTS[v][0]}")
ax.axhline(REF_SHUTDOWN, color="black", ls=":", lw=1.5, label=f"참고: 실제 휴가 평일 실측 {REF_SHUTDOWN:.1f} kW")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['날짜'][5:]}({r['요일']})" for _, r in scen.iterrows()], fontsize=8)
ax.set_ylabel("예측 일평균 (kW)")
ax.set_title("가정: 9월 평일을 '계획 휴무(생산 0)'로 바꿨을 때의 예측 (9/1 이전 학습 모델)", fontsize=11)
ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIG / "round1_fig2_whatif_weekday_shutdown.png", dpi=150)

meta = {
    "chosen": CHOSEN, "passed": passed, "noise_mae": NOISE_MAE, "noise_peak": NOISE_PEAK,
    "r_changed_rows_oof": int(_chg.sum()), "ref_shutdown_kw": REF_SHUTDOWN,
    "gate_splits_is_shutdown_full_fit": {v: int(full_fit[v]["_gate_splits"].get("is_shutdown", 0)) for v in VARIANTS},
    "total_sec": round(time.perf_counter() - T0, 1),
}
(OUT / "meta.json").write_bytes(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
print("\n== OOF 판정 ==")
print(tbl[["ID", "OOF MAE", "ΔOOF MAE", "ΔMAE CI 하한", "ΔMAE CI 상한", "Δfold2", "Δfold5", "Δ정상일 MAE",
           "ΔRecall", "ΔPeak-MAE", "Δ이틀 제외 MAE", "ΔD2 OOF MAE", "모두 통과"]].round(3).to_string(index=False))
print("\n== 테스트 영향 ==")
print(test_tbl.round(4).to_string(index=False))
print("\n== 가정 시나리오 ==")
print(scen.round(1).to_string(index=False))
print(f"\n[완료] {meta['total_sec']}s · 채택 {CHOSEN}")
