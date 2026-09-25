"""사후 가설 D — 학습 데이터에서 '증강 불일치일'을 빼면 게이트의 지름길이 사라지나. **판정에 쓰지 않는다.**

사후 진단(posthoc_gate_fix.py §8)에서, fold3 게이트를 증강 불일치일(생산 0 인데 전력이 가동 수준으로 덮어써진 날,
2/11·3/1 등 9일) 없이 학습하면 7/31·8/2·8/3 이 규칙 없이도 기저로 분류됐다. 이것이 전 fold·전 시나리오에서도
성립하는지, 특히 R·G 가 실패한 ERP 결측일(7/13·7/15, 실제 가동)에서 안전한지를 본다.

D = 레짐 모델 구조·하이퍼파라미터는 현행 그대로, **학습 행에서 증강 불일치일만 제외**(예측 단계 규칙 없음).
증강 불일치 여부는 과거 날의 기록(생산 0·전력 가동 수준)으로 정하므로 학습 데이터 정제에만 쓰고 예측에는 쓰지 않는다.

실행 (저장소 루트에서):
    PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiments/2026-09-25_gate_fix/posthoc_datafix.py
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
(WORK / "data").mkdir(parents=True, exist_ok=True)
if not (WORK / "data" / "okm_augumented_2021.csv").exists():
    shutil.copy2(REPO / "data" / "okm_augumented_2021.csv", WORK / "data" / "okm_augumented_2021.csv")
os.chdir(WORK)
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T0 = time.perf_counter()
with contextlib.redirect_stdout(io.StringIO()):
    import matplotlib

    matplotlib.use("Agg")
    import s00_env as s00  # noqa: E402
    import s01_diagnose as s01  # noqa: E402
    import s02_features as s02  # noqa: E402
    import s03_split as s03  # noqa: E402
import lightgbm as lgb  # noqa: E402

WANT = {"LGB_BASE", "N_TRIALS", "N_ESTIMATORS", "_TIMING", "run_cv", "REGIME_CUT", "make_regime_model"}
ns: dict = {}
for _m in (s00, s01, s02, s03):
    ns.update(vars(_m))
ns.update(time=time, lgb=lgb)
_nodes = [n for n in ast.parse((REPO / "src" / "s05_models.py").read_text(encoding="utf-8")).body
          if (isinstance(n, ast.FunctionDef) and n.name in WANT)
          or (isinstance(n, ast.Assign) and {t.id for t in n.targets if isinstance(t, ast.Name)} & WANT)
          or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in WANT)]
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "s05_models.py", "exec"), ns)
run_cv, make_regime_model = ns["run_cv"], ns["make_regime_model"]
assert ns["N_ESTIMATORS"] == 800

THETA = s01.THETA
CAL = s01.operating_calendar
FEATURE_COLS = list(s02.FEATURE_COLS)
ACTIVE_FOLDS = s03.ACTIVE_FOLDS
AUG_DAYS = CAL.index[CAL["is_aug_inconsistent"]]
POST = pd.read_csv(OUT / "posthoc_oof_predictions.csv", encoding="utf-8-sig", parse_dates=["ts"]).set_index("ts")
TAU_A = float(json.loads((OUT / "posthoc_meta.json").read_bytes())["tau_A"])


def make_datafix():
    """현행 레짐 모델을 증강 불일치일을 뺀 학습 행으로 적합한다."""
    inner = make_regime_model(3)

    def fit_fn(Xtr, ytr, Xva):
        keep = ~Xtr.index.normalize().isin(AUG_DAYS)
        return inner(Xtr[keep], ytr[keep], Xva)

    return fit_fn


def predict_from_artifacts(art: dict, X: pd.DataFrame, tgt: str) -> np.ndarray:
    """적합된 레짐 모델(_artifacts)로 새 피처 행을 예측한다(s05 predict 와 같은 규칙, 재적합 없음)."""
    rr = art["gate"].predict(X[art["gcols"]])
    out = np.empty(len(X), float)
    assigned = np.zeros(len(X), bool)
    for r, reg in art["regs"][tgt].items():
        m = rr == r
        if m.any():
            out[m] = reg.predict(X[m])
            assigned |= m
    if (~assigned).any():
        out[~assigned] = art["fallbacks"][tgt].predict(X[~assigned])
    return out


print(f"[D] 증강 불일치일 {len(AUG_DAYS)}일: {', '.join(d.strftime('%m-%d') for d in AUG_DAYS)}", flush=True)
rows = []
res = {}
for cond in ("D1", "D2"):
    res[(cond, "A")] = run_cv("D-base", make_regime_model(3), cond)
    res[(cond, "D")] = run_cv("D-datafix", make_datafix(), cond)
a, dd = res[("D1", "A")]["oof"], res[("D1", "D")]["oof"]
# CSV 왕복은 마지막 자리 반올림이 있어 허용 오차로 비교한다
assert np.allclose(a["pred_avg"].to_numpy(), POST["pred_avg[A]"].to_numpy(), rtol=0, atol=1e-9)
y = a["y_avg"].to_numpy()
pa, pd_ = a["pred_avg"].to_numpy(), dd["pred_avg"].to_numpy()
shut = POST["is_shutdown"].to_numpy() == 1
days = pd.Index(a.index).normalize()
top3 = pd.to_datetime(["2021-07-31", "2021-08-02", "2021-08-03"])

summary = {
    "OOF MAE A": float(np.abs(y - pa).mean()), "OOF MAE D": float(np.abs(y - pd_).mean()),
    "D2 OOF MAE A": float(s03.mae(res[("D2", "A")]["oof"]["y_avg"], res[("D2", "A")]["oof"]["pred_avg"])),
    "D2 OOF MAE D": float(s03.mae(res[("D2", "D")]["oof"]["y_avg"], res[("D2", "D")]["oof"]["pred_avg"])),
    "비휴무 MAE A": float(np.abs(y - pa)[~shut].mean()), "비휴무 MAE D": float(np.abs(y - pd_)[~shut].mean()),
    "문제 3일 MAE A": float(np.abs(y - pa)[days.isin(top3)].mean()), "문제 3일 MAE D": float(np.abs(y - pd_)[days.isin(top3)].mean()),
    "문제 3일 제외 MAE A": float(np.abs(y - pa)[~days.isin(top3)].mean()),
    "문제 3일 제외 MAE D": float(np.abs(y - pd_)[~days.isin(top3)].mean()),
}
for f in ACTIVE_FOLDS:
    m = a["fold"].to_numpy() == f
    summary[f"fold{f} A"] = float(np.abs(y - pa)[m].mean())
    summary[f"fold{f} D"] = float(np.abs(y - pd_)[m].mean())
    summary[f"비휴무 fold{f} Δ"] = float(np.abs(y - pd_)[m & ~shut].mean() - np.abs(y - pa)[m & ~shut].mean())
for lab, tau in (("τ_A 고정", TAU_A), ("자기 τ", res[("D1", "D")]["tau"])):
    c = s03.classification_metrics(dd["y_cls"], (dd["pred_peak"] >= tau).astype(int), dd["pred_peak"])
    summary.update({f"D {lab} TP": c["TP"], f"D {lab} FN": c["FN"], f"D {lab} FP": c["FP"], f"D {lab} F1": c["F1"]})
# 부트스트랩 (s03 과 같은 절차)
bs = s03.paired_bootstrap_mae(y, pa, pd_, index=a.index)
summary.update({"ΔMAE(D−A)": -bs["diff"], "ΔMAE CI 하한": -bs["ci_high"], "ΔMAE CI 상한": -bs["ci_low"], "p": bs["p_value"]})
# 날짜별 부호표 요약
dae = np.abs(y - pd_) - np.abs(y - pa)
by_day = pd.Series(dae, index=days).groupby(level=0).sum()
summary.update({"바뀐 날": int((by_day != 0).sum()), "좋아진 날": int((by_day < 0).sum()), "나빠진 날": int((by_day > 0).sum()),
                "나빠진 날 최대(kWh)": float(by_day.max())})
worst = by_day.sort_values(ascending=False).head(5)

# ERP 결측일 하방 시험 (posthoc_gate_fix.py §6 과 같은 방식)
ERP_DAYS = pd.to_datetime(["2021-07-13", "2021-07-15"])
dfm = s01.df.copy()
on = dfm.index.normalize().isin(ERP_DAYS)
dfm.loc[on, "is_erp_missing"] = False
calm = s01.build_operating_calendar(dfm)
with contextlib.redirect_stdout(io.StringIO()):
    featm = s02.build_features(dfm, calm, THETA)
Xe = featm.loc[on, FEATURE_COLS]
ye = s01.df.loc[on, ["y_avg", "y_peak"]]
Xtr2, ytr2, _, _, _ = s03.get_fold_data(2, "D1")
down = []
for tag, fit in (("A", make_regime_model(3)), ("D", make_datafix())):
    o = fit(Xtr2, ytr2, Xe)
    for d in ERP_DAYS:
        m = Xe.index.normalize() == d
        down.append({"ID": tag, "날짜": d.strftime("%Y-%m-%d"), "실측 일평균": float(ye["y_avg"][m].mean()),
                     "예측 일평균": float(o["pred_avg"][m].mean()),
                     "MAE": float(np.abs(ye["y_avg"].to_numpy()[m] - o["pred_avg"][m]).mean()),
                     "실제 피크 시간": int((ye["y_peak"].to_numpy()[m] >= THETA).sum()),
                     "경보 시간(≥τ_A)": int((o["pred_peak"][m] >= TAU_A).sum()),
                     "적중 경보": int(((o["pred_peak"][m] >= TAU_A) & (ye["y_peak"].to_numpy()[m] >= THETA)).sum())})
down_tbl = pd.DataFrame(down)

# 가정 시나리오 + 테스트 (전체 학습)
Xtr_f, ytr_f, Xte, yte, _ = s03.get_test_data("D1")
full = {"A": make_regime_model(3)(Xtr_f, ytr_f, Xte), "D": make_datafix()(Xtr_f, ytr_f, Xte)}
test_rows = []
for tag, o in full.items():
    c = s03.classification_metrics(yte["y_cls"], (o["pred_peak"] >= TAU_A).astype(int), o["pred_peak"])
    test_rows.append({"ID": tag, "테스트 MAE": float(s03.mae(yte["y_avg"], o["pred_avg"])), "TP(τ_A)": c["TP"],
                      "FN(τ_A)": c["FN"], "FP(τ_A)": c["FP"], "F1(τ_A)": c["F1"]})
test_tbl = pd.DataFrame(test_rows)
# 가정 시나리오: 전체 학습 게이트를 쓰려면 _predict 가 필요하다 → 전체 학습 적합을 가정 행에 다시 적용
scen = []
weekdays = [d for d in pd.date_range("2021-09-01", "2021-09-14", freq="D") if d.dayofweek < 5]
for d in weekdays:
    dfm_ = s01.df.copy()
    on_d = dfm_.index.normalize() == d
    dfm_.loc[on_d, ["생산량", "공장인원"]] = 0
    calm_ = s01.build_operating_calendar(dfm_)
    with contextlib.redirect_stdout(io.StringIO()):
        featm_ = s02.build_features(dfm_, calm_, THETA)
    Xd = featm_.loc[on_d, FEATURE_COLS]
    for tag, o in full.items():
        pa_d = predict_from_artifacts(o["_artifacts"], Xd, "y_avg")
        pp_d = predict_from_artifacts(o["_artifacts"], Xd, "y_peak")
        scen.append({"날짜": d.strftime("%Y-%m-%d"), "ID": tag, "예측 일평균": float(pa_d.mean()),
                     "경보 시간(≥τ_A)": int((pp_d >= TAU_A).sum())})
scen_tbl = pd.DataFrame(scen).pivot(index="날짜", columns="ID")

pd.DataFrame([summary]).to_csv(OUT / "posthoc_datafix_summary.csv", index=False, encoding="utf-8-sig")
down_tbl.to_csv(OUT / "posthoc_datafix_erp_downside.csv", index=False, encoding="utf-8-sig")
test_tbl.to_csv(OUT / "posthoc_datafix_test.csv", index=False, encoding="utf-8-sig")
scen_tbl.to_csv(OUT / "posthoc_datafix_whatif.csv", encoding="utf-8-sig")
pd.DataFrame({"날짜": worst.index.strftime("%Y-%m-%d"), "Δ오차 합(kWh)": worst.to_numpy()}).to_csv(
    OUT / "posthoc_datafix_worst_days.csv", index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
for k, v in summary.items():
    print(f"  {k:<22s} {v:.4f}" if isinstance(v, float) else f"  {k:<22s} {v}")
print("\n== 가장 나빠진 날 (D − A, kWh) ==")
print(worst.round(2).to_string())
print("\n== ERP 결측 하방 ==")
print(down_tbl.round(3).to_string(index=False))
print("\n== 테스트 (보고만) ==")
print(test_tbl.round(4).to_string(index=False))
print("\n== 가정 시나리오 ==")
print(scen_tbl.round(1).to_string())
print(f"\n[완료] {time.perf_counter() - T0:.0f}s")
