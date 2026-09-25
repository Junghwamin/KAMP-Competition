"""사후 진단 2 — 검증 워크플로 지적 대응. **판정에 쓰지 않는다.**

1. 평일 공휴일 + 계획 0 인 날(1/1·2/12)은 실측 46~47 kW(중간 레짐)였다. 추석형 날에 R(기저 강제)이 과소예측하는지 본다.
   → 이 날들은 모든 모델의 학습 데이터 안에 있으므로(in-sample) 방향만 참고한다.
2. 약한 보정 R-soft: 확정 휴무일에 게이트가 '가동(2)'을 고르면 기저·중간 중 확률이 큰 쪽으로 보낸다(가동만 막는다).
   문제 3일(OOF, fold3)과 1/1·2/12(in-sample), ERP 결측일(fold2 학습창)에서 R 과 비교한다.
3. ERP 결측일의 시간별 실측 피크·경보(그림 fig3 표시용).

실행 (저장소 루트에서):
    PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiments/2026-09-25_gate_fix/posthoc_holiday_soft.py
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

with contextlib.redirect_stdout(io.StringIO()):
    import matplotlib

    matplotlib.use("Agg")
    import s00_env as s00  # noqa: E402
    import s01_diagnose as s01  # noqa: E402
    import s02_features as s02  # noqa: E402
    import s03_split as s03  # noqa: E402
import lightgbm as lgb  # noqa: E402

WANT = {"LGB_BASE", "N_TRIALS", "N_ESTIMATORS", "_TIMING", "REGIME_CUT", "make_regime_model"}
ns: dict = {}
for _m in (s00, s01, s02, s03):
    ns.update(vars(_m))
ns.update(time=time, lgb=lgb)
_nodes = [n for n in ast.parse((REPO / "src" / "s05_models.py").read_text(encoding="utf-8")).body
          if (isinstance(n, ast.FunctionDef) and n.name in WANT)
          or (isinstance(n, ast.Assign) and {t.id for t in n.targets if isinstance(t, ast.Name)} & WANT)
          or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in WANT)]
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "s05_models.py", "exec"), ns)
make_regime_model = ns["make_regime_model"]
assert ns["N_ESTIMATORS"] == 800
THETA = s01.THETA
FEATURE_COLS = list(s02.FEATURE_COLS)
TAU_A = float(json.loads((OUT / "posthoc_meta.json").read_bytes())["tau_A"])


def predict_mode(art: dict, X: pd.DataFrame, tgt: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """mode: 'A' 현행 · 'R' 휴무면 기저 강제 · 'Rsoft' 휴무면 가동(2)만 막고 기저·중간 중 확률이 큰 쪽."""
    gate, classes = art["gate"], np.asarray(art["classes"])
    proba = gate.predict_proba(X[art["gcols"]])
    rr = classes[np.argmax(proba, axis=1)]
    shut = X["is_shutdown"].to_numpy() == 1
    if mode == "R":
        rr = np.where(shut, 0, rr)
    elif mode == "Rsoft":
        non_op = [i for i, c in enumerate(classes) if c != 2]
        alt = classes[np.asarray(non_op)[np.argmax(proba[:, non_op], axis=1)]]
        rr = np.where(shut & (rr == 2), alt, rr)
    out = np.empty(len(X), float)
    done = np.zeros(len(X), bool)
    for r, reg in art["regs"][tgt].items():
        m = rr == r
        if m.any():
            out[m] = reg.predict(X[m])
            done |= m
    if (~done).any():
        out[~done] = art["fallbacks"][tgt].predict(X[~done])
    return out, rr


rows, hourly = [], []
fit = make_regime_model(3)


def evaluate(tag: str, art: dict, X: pd.DataFrame, y: pd.DataFrame, days):
    for mode in ("A", "R", "Rsoft"):
        pa, rr = predict_mode(art, X, "y_avg", mode)
        pp, _ = predict_mode(art, X, "y_peak", mode)
        for d in days:
            m = X.index.normalize() == d
            yp = y["y_peak"].to_numpy()[m]
            rows.append({"사례": tag, "날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금토일"[d.dayofweek], "방식": mode,
                         "실측 일평균": float(y["y_avg"].to_numpy()[m].mean()), "예측 일평균": float(pa[m].mean()),
                         "MAE": float(np.abs(y["y_avg"].to_numpy()[m] - pa[m]).mean()),
                         "레짐 분포": " ".join(f"{int(k)}:{int(v)}" for k, v in pd.Series(rr[m]).value_counts().sort_index().items()),
                         "실제 피크 시간": int((yp >= THETA).sum()), "경보 시간(≥τ_A)": int((pp[m] >= TAU_A).sum()),
                         "적중 경보": int(((pp[m] >= TAU_A) & (yp >= THETA)).sum())})
            if tag.startswith("ERP"):
                for k, t in enumerate(X.index[m]):
                    hourly.append({"ts": t, "방식": mode, "y_avg": float(y["y_avg"].to_numpy()[m][k]),
                                   "y_peak": float(yp[k]), "pred_avg": float(pa[m][k]), "pred_peak": float(pp[m][k])})


# 1) 문제 3일 (OOF, fold3 학습)
Xtr3, ytr3, Xva3, yva3, _ = s03.get_fold_data(3, "D1")
art3 = fit(Xtr3, ytr3, Xva3)["_artifacts"]
evaluate("문제일(OOF)", art3, Xva3, yva3, pd.to_datetime(["2021-07-31", "2021-08-02", "2021-08-03"]))

# 2) 평일 공휴일 + 계획 0 (in-sample: 9/1 이전 전체 학습 모델) — 1/1 은 워밍업이라 피처 결측이 있을 수 있다
Xtr_f, ytr_f, _, _, _ = s03.get_test_data("D1")
art_f = fit(Xtr_f, ytr_f, Xtr_f.iloc[:5])["_artifacts"]
hol_days = [d for d in pd.to_datetime(["2021-01-01", "2021-02-12"])
            if not s02.feat.loc[s02.feat.index.normalize() == d, FEATURE_COLS].isna().any().any()]
if hol_days:
    mh = s02.feat.index.normalize().isin(hol_days)
    evaluate("평일 공휴일(학습 내)", art_f, s02.feat.loc[mh, FEATURE_COLS], s02.feat.loc[mh, ["y_avg", "y_peak"]], hol_days)

# 3) ERP 결측 가동일 (fold2 학습창, is_erp_missing 을 원점 기준 False 로)
ERP_DAYS = pd.to_datetime(["2021-07-13", "2021-07-15"])
dfm = s01.df.copy()
on = dfm.index.normalize().isin(ERP_DAYS)
dfm.loc[on, "is_erp_missing"] = False
calm = s01.build_operating_calendar(dfm)
with contextlib.redirect_stdout(io.StringIO()):
    featm = s02.build_features(dfm, calm, THETA)
Xtr2, ytr2, _, _, _ = s03.get_fold_data(2, "D1")
art2 = fit(Xtr2, ytr2, Xtr2.iloc[:5])["_artifacts"]
evaluate("ERP 결측 가동일", art2, featm.loc[on, FEATURE_COLS], s01.df.loc[on, ["y_avg", "y_peak"]], ERP_DAYS)

tbl = pd.DataFrame(rows)
tbl.to_csv(OUT / "posthoc_holiday_soft.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(hourly).to_csv(OUT / "posthoc_erp_missing_hourly_peaks.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 220)
print("평일 공휴일 검사 대상:", [d.strftime("%m-%d") for d in hol_days])
print(tbl.round(2).to_string(index=False))
