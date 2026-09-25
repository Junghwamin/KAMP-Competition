"""사후 진단(post-hoc) — 1차 판정(run_gate_fix.py) 뒤 검토에서 나온 질문에 답한다. **판정에는 쓰지 않는다.**

1. τ 를 현행값(τ_A)으로 고정했을 때의 피크 탐지 (④ 불합격이 τ 재조정 탓인지)
2. R 의 부트스트랩 분해 (① 불합격이 '효과 0' 표본 때문인지)
3. 바뀐 행의 날짜별 부호표 (어디가 좋아지고 어디가 나빠졌나)
4. 휴무가 아닌 행의 MAE 변화 (해를 끼치는지 — R 은 구조상 0)
5. 게이트 전용 잡음 기준선 (G 에 맞는 잡음 폭)
6. **ERP 결측일 하방 시험** — 7/13·7/15 는 생산량 0 으로 기록됐지만 실제로는 가동(136 kW)했다.
   서비스에서는 원점에 이 사실을 모르므로 계획 휴무로 보인다. 그때 R 이 무엇을 하나
7. 가정 시나리오 확장 — 월요일/화~금 분리, 공휴일 표시, 연속 3일 휴무(추석형)
8. 근본 원인 — fold3 학습의 평일 계획 휴무 예시, 증강 불일치일을 빼고 게이트를 학습하면?
+ 전 변형의 OOF·테스트 예측 저장 (감사 추적)

실행 (저장소 루트에서):
    PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiments/2026-09-25_gate_fix/posthoc_gate_fix.py
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
_csv = WORK / "data" / "okm_augumented_2021.csv"
if not _csv.exists():
    shutil.copy2(REPO / "data" / "okm_augumented_2021.csv", _csv)
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

assert s00.FAST is False and Path(s00.OUTPUT_DIR).resolve().parent == WORK.resolve()
WANT = {"LGB_BASE", "N_TRIALS", "N_ESTIMATORS", "_TIMING", "run_cv", "REGIME_CUT"}
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
run_cv = ns["run_cv"]
LGB_BASE, N_EST, REGIME_CUT = ns["LGB_BASE"], ns["N_ESTIMATORS"], ns["REGIME_CUT"]
assert N_EST == 800

THETA = s01.THETA
CAL = s01.operating_calendar
FG = s02.FEATURE_GROUPS
FEATURE_COLS = list(s02.FEATURE_COLS)
ACTIVE_FOLDS = s03.ACTIVE_FOLDS
TOP_DAYS = pd.to_datetime(["2021-08-02", "2021-08-03"])
GATE_22 = FG["시간정보"] + FG["공정상태"] + FG["생산정보"]
ROUND1 = pd.read_csv(OUT / "gate_fix_oof.csv", encoding="utf-8-sig").set_index("ID")


def make_regime_model_v(gate_features=None, shutdown_override=False, reg_features=None):
    """run_gate_fix.py 와 같은 모델 (reg_features 만 추가: 게이트 전용 잡음 기준선용)."""

    def regime_label(y_avg):
        return np.digitize(np.asarray(y_avg), [30.0, REGIME_CUT])

    def fit_fn(Xtr, ytr, Xva):
        gcols = [c for c in (gate_features or Xtr.columns) if c in Xtr.columns]
        rcols = list(reg_features) if reg_features is not None else list(Xtr.columns)
        rtr = regime_label(ytr["y_avg"])
        gate = lgb.LGBMClassifier(n_estimators=400, **{**LGB_BASE, "objective": "multiclass"})
        gate.fit(Xtr[gcols], rtr)
        regs, fallbacks = {}, {}
        for tgt in ("y_avg", "y_peak"):
            regs[tgt] = {}
            for r in np.unique(rtr):
                m_tr = rtr == r
                if m_tr.sum() < 20:
                    continue
                reg = lgb.LGBMRegressor(n_estimators=N_EST, **LGB_BASE)
                reg.fit(Xtr.loc[m_tr, rcols], ytr[tgt][m_tr])
                regs[tgt][int(r)] = reg
            fb = lgb.LGBMRegressor(n_estimators=N_EST, **LGB_BASE)
            fb.fit(Xtr[rcols], ytr[tgt])
            fallbacks[tgt] = fb

        def route(X):
            rr = gate.predict(X[gcols])
            if shutdown_override:
                rr = np.where(X["is_shutdown"].to_numpy() == 1, 0, rr)
            return rr

        def predict(X, tgt):
            rr = route(X)
            out = np.empty(len(X), float)
            assigned = np.zeros(len(X), bool)
            for r, reg in regs[tgt].items():
                m = rr == r
                if m.any():
                    out[m] = reg.predict(X.loc[m, rcols])
                    assigned |= m
            if (~assigned).any():
                out[~assigned] = fallbacks[tgt].predict(X.loc[~assigned, rcols])
            return out

        return {"pred_avg": predict(Xva, "y_avg"), "pred_peak": predict(Xva, "y_peak"),
                "_route": route, "_predict": predict, "_gate": gate}

    return fit_fn


VARIANTS = {
    "A": ("현행", dict()),
    "R": ("계획 휴무 강제", dict(shutdown_override=True)),
    "G": ("게이트 입력 제한(22개)", dict(gate_features=GATE_22)),
    "RG": ("R + G", dict(gate_features=GATE_22, shutdown_override=True)),
}


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


def cls_at(oof: pd.DataFrame, tau: float) -> dict:
    return s03.classification_metrics(oof["y_cls"], (oof["pred_peak"] >= tau).astype(int), oof["pred_peak"])


# ══ 0. 1차와 같은 결과를 다시 만들고 예측을 저장 ═══════════════════════════════════
print("[0] 교차검증 재실행 (1차와 같아야 함) + 예측 저장", flush=True)
d1 = {v: run_cv(f"posthoc-{v}", make_regime_model_v(**VARIANTS[v][1]), "D1") for v in VARIANTS}
for v in VARIANTS:
    m = s03.mae(d1[v]["oof"]["y_avg"], d1[v]["oof"]["pred_avg"])
    assert abs(m - ROUND1.loc[v, "OOF MAE"]) < 1e-9, (v, m)
base = d1["A"]["oof"]
wide = base[["fold", "y_avg", "y_peak", "y_cls"]].copy()
wide["is_shutdown"] = s02.feat.loc[base.index, "is_shutdown"].to_numpy()
for v in VARIANTS:
    wide[f"pred_avg[{v}]"] = d1[v]["oof"]["pred_avg"].to_numpy()
    wide[f"pred_peak[{v}]"] = d1[v]["oof"]["pred_peak"].to_numpy()
wide.index.name = "ts"
wide.reset_index().to_csv(OUT / "posthoc_oof_predictions.csv", index=False, encoding="utf-8-sig")

# ══ 1. τ 고정 ═════════════════════════════════════════════════════════════════
print("[1] τ_A 고정 피크 탐지", flush=True)
TAU_A = d1["A"]["tau"]
rows = []
for v in VARIANTS:
    oof = d1[v]["oof"]
    for label, tau in (("τ_A 고정", TAU_A), ("자기 τ", d1[v]["tau"])):
        m = cls_at(oof, tau)
        rows.append({"ID": v, "기준 τ": label, "τ": tau, "TP": m["TP"], "FN": m["FN"], "FP": m["FP"],
                     "Recall": m["Recall"], "Precision": m["Precision"], "F1": m["F1"]})
    fold_tau = d1[v]["fold_metrics"].set_index("fold")["τ"]
    rows[-1].update({f"fold{f} τ": float(fold_tau[f]) for f in ACTIVE_FOLDS})
tau_tbl = pd.DataFrame(rows)
save(tau_tbl, "posthoc_fixed_tau")

# ══ 2. 부트스트랩 분해 (s03.paired_bootstrap_mae 와 같은 절차·시드) ══════════════════
print("[2] 부트스트랩 분해", flush=True)
y = base["y_avg"].to_numpy()
days = pd.Index(base.index).normalize()
blocks = [np.where(days == d)[0] for d in days.unique()]
boot_rows = []
for v in ("R", "G", "RG"):
    ea = np.abs(y - base["pred_avg"].to_numpy())
    eb = np.abs(y - d1[v]["oof"]["pred_avg"].to_numpy())
    rng = np.random.default_rng(42)
    stat = np.empty(2000)
    for i in range(2000):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[j] for j in pick])
        stat[i] = ea[idx].mean() - eb[idx].mean()          # 기준 − 변형 (양수 = 변형이 나음)
    changed_days = int(pd.Series(ea != eb, index=days).groupby(level=0).any().sum())
    boot_rows.append({"ID": v, "바뀐 날 수": changed_days, "표본 수": 2000,
                      "변형이 나은 표본": int((stat > 0).sum()), "차이 0 표본": int((stat == 0).sum()),
                      "변형이 나쁜 표본": int((stat < 0).sum()),
                      "p(1차 방식)": float(min(1.0, 2 * min((stat >= 0).mean(), (stat <= 0).mean())))})
boot_tbl = pd.DataFrame(boot_rows)
save(boot_tbl, "posthoc_bootstrap")

# ══ 3·4. 바뀐 행의 날짜별 부호표 · 휴무가 아닌 행의 MAE ══════════════════════════════
print("[3] 날짜별 부호표 · 비휴무 행 MAE", flush=True)
sign_rows, harm_rows = [], []
shut = wide["is_shutdown"].to_numpy() == 1
for v in ("R", "G", "RG"):
    pv = d1[v]["oof"]["pred_avg"].to_numpy()
    pa = base["pred_avg"].to_numpy()
    chg = pv != pa
    dae = np.abs(y - pv) - np.abs(y - pa)
    dfc = pd.DataFrame({"날짜": days[chg], "Δ|오차|": dae[chg], "휴무": shut[chg]})
    for d, g in dfc.groupby("날짜"):
        sign_rows.append({"ID": v, "날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금토일"[d.dayofweek],
                          "계획 휴무": bool(g["휴무"].all()), "바뀐 행": len(g),
                          "좋아진 행": int((g["Δ|오차|"] < 0).sum()), "나빠진 행": int((g["Δ|오차|"] > 0).sum()),
                          "Δ오차 합(kWh)": float(g["Δ|오차|"].sum())})
    for scope, mask in (("비휴무 행 전체", ~shut), *[(f"비휴무 fold{f}", (~shut) & (wide["fold"].to_numpy() == f)) for f in ACTIVE_FOLDS]):
        harm_rows.append({"ID": v, "범위": scope, "행 수": int(mask.sum()),
                          "현행 MAE": float(np.abs(y - pa)[mask].mean()), "변형 MAE": float(np.abs(y - pv)[mask].mean()),
                          "Δ": float(np.abs(y - pv)[mask].mean() - np.abs(y - pa)[mask].mean())})
sign_tbl = pd.DataFrame(sign_rows)
harm_tbl = pd.DataFrame(harm_rows)
save(sign_tbl, "posthoc_changed_rows_by_date")
save(harm_tbl, "posthoc_nonshutdown_harm")

# ══ 5. 게이트 전용 잡음 기준선 ══════════════════════════════════════════════════
print("[5] 게이트 전용 잡음 기준선 (5회)", flush=True)
null_rows = []
orig_feat = s03.feat
for k, seed in enumerate((2000, 2001, 2002, 2003, 2004)):
    f = orig_feat.copy()
    f["noise"] = np.random.default_rng(seed).standard_normal(len(f))
    s03.feat = f
    try:
        r = run_cv(f"posthoc-NULLg{k}", make_regime_model_v(gate_features=[*FEATURE_COLS, "noise"],
                                                            reg_features=FEATURE_COLS), "D1",
                   features=[*FEATURE_COLS, "noise"])
    finally:
        s03.feat = orig_feat
    pn = r["oof"]["pred_avg"].to_numpy()
    pa = base["pred_avg"].to_numpy()
    row = {"시드": seed, "ΔOOF MAE": float(np.abs(y - pn).mean() - np.abs(y - pa).mean()),
           "Δ비휴무 MAE": float(np.abs(y - pn)[~shut].mean() - np.abs(y - pa)[~shut].mean())}
    for fo in ACTIVE_FOLDS:
        mf = wide["fold"].to_numpy() == fo
        row[f"Δfold{fo}"] = float(np.abs(y - pn)[mf].mean() - np.abs(y - pa)[mf].mean())
    null_rows.append(row)
null_tbl = pd.DataFrame(null_rows)
save(null_tbl, "posthoc_gate_null")

# ══ 전체 학습 모델 (테스트·가정 시나리오용) ════════════════════════════════════════
Xtr_f, ytr_f, Xte, yte, idx_te = s03.get_test_data("D1")
full = {v: make_regime_model_v(**VARIANTS[v][1])(Xtr_f, ytr_f, Xte) for v in VARIANTS}
te = pd.DataFrame({"ts": idx_te, "y_avg": yte["y_avg"].to_numpy(), "y_peak": yte["y_peak"].to_numpy()})
for v in VARIANTS:
    te[f"pred_avg[{v}]"] = full[v]["pred_avg"]
    te[f"pred_peak[{v}]"] = full[v]["pred_peak"]
te.to_csv(OUT / "posthoc_test_predictions.csv", index=False, encoding="utf-8-sig")

# ══ 6. ERP 결측일 하방 시험 (7/13·7/15) ═════════════════════════════════════════
print("[6] ERP 결측일 하방 시험", flush=True)
ERP_DAYS = pd.to_datetime(["2021-07-13", "2021-07-15"])
dfm = s01.df.copy()
on = dfm.index.normalize().isin(ERP_DAYS)
assert dfm.loc[on, "is_erp_missing"].all() and (dfm.loc[on, "생산량"] == 0).all()
dfm.loc[on, "is_erp_missing"] = False          # 원점에서는 ERP 결측을 알 수 없다(서빙과 같게)
calm = s01.build_operating_calendar(dfm)
assert calm.loc[ERP_DAYS, "is_shutdown"].all()
with contextlib.redirect_stdout(io.StringIO()):
    featm = s02.build_features(dfm, calm, THETA)
Xe = featm.loc[on, FEATURE_COLS]
ye = s01.df.loc[on, ["y_avg", "y_peak"]]
Xtr2, ytr2, _, _, _ = s03.get_fold_data(2, "D1")     # 7/13·7/15 는 fold2 검증구간 — 그 학습창으로 적합
down_rows, down_hourly = [], pd.DataFrame({"ts": Xe.index, "y_avg": ye["y_avg"].to_numpy()})
for v in VARIANTS:
    o = make_regime_model_v(**VARIANTS[v][1])(Xtr2, ytr2, Xe)
    rr = o["_route"](Xe)
    down_hourly[f"pred[{v}]"] = o["pred_avg"]
    for d in ERP_DAYS:
        m = Xe.index.normalize() == d
        down_rows.append({"ID": v, "날짜": d.strftime("%Y-%m-%d"), "실측 일평균": float(ye["y_avg"][m].mean()),
                          "예측 일평균": float(o["pred_avg"][m].mean()),
                          "MAE": float(np.abs(ye["y_avg"].to_numpy()[m] - o["pred_avg"][m]).mean()),
                          "실제 피크 시간(≥θ)": int((ye["y_peak"].to_numpy()[m] >= THETA).sum()),
                          "경보 시간(≥τ_A)": int((o["pred_peak"][m] >= TAU_A).sum()),
                          "적중 경보": int(((o["pred_peak"][m] >= TAU_A) & (ye["y_peak"].to_numpy()[m] >= THETA)).sum()),
                          "08~17시 가동 배정": int(((rr == 2) & (Xe.index.hour >= 8) & (Xe.index.hour <= 17))[m].sum())})
down_tbl = pd.DataFrame(down_rows)
save(down_tbl, "posthoc_erp_missing_downside")
down_hourly.to_csv(OUT / "posthoc_erp_missing_hourly.csv", index=False, encoding="utf-8-sig")

# ══ 7. 가정 시나리오 확장 ══════════════════════════════════════════════════════
print("[7] 가정 시나리오 확장", flush=True)
POWER_HISTORY = [c for c in FEATURE_COLS if c in FG["과거전력"] + FG["이동통계"]]
CHECK_SAME = POWER_HISTORY + FG["기상정보"] + [c for c in FG["시간정보"] if c != "is_holiday"]


def scenario_rows(tag, target_days, dfm_, holidays):
    calm_ = s01.build_operating_calendar(dfm_)
    with contextlib.redirect_stdout(io.StringIO()):
        featm_ = s02.build_features(dfm_, calm_, THETA, holidays=holidays)
    out_rows = []
    for d in target_days:
        m = featm_.index.normalize() == d
        Xd = featm_.loc[m, FEATURE_COLS]
        assert (Xd["is_shutdown"] == 1).all()
        for v in ("A", "R", "G"):
            pa = full[v]["_predict"](Xd, "y_avg")
            pp = full[v]["_predict"](Xd, "y_peak")
            rr = full[v]["_route"](Xd)
            work = (Xd.index.hour >= 8) & (Xd.index.hour <= 17)
            out_rows.append({"시나리오": tag, "날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금토일"[d.dayofweek],
                             "휴무 n일차": int(Xd["shutdown_nth"].iloc[0]), "공휴일 표시": int(Xd["is_holiday"].iloc[0]),
                             "ID": v, "예측 일평균": float(pa.mean()), "08~17시 가동 배정": int((rr[work] == 2).sum()),
                             "경보 시간(≥τ_A)": int((pp >= TAU_A).sum())})
    return featm_, out_rows


scen_rows = []
weekdays = [d for d in pd.date_range("2021-09-01", "2021-09-14", freq="D") if d.dayofweek < 5]
for hol_on in (False, True):
    for d in weekdays:
        dfm_ = s01.df.copy()
        on_d = dfm_.index.normalize() == d
        dfm_.loc[on_d, ["생산량", "공장인원"]] = 0
        hol = s02.HOLIDAYS_2021.union(pd.DatetimeIndex([d])) if hol_on else None
        featm_, rws = scenario_rows("단일 평일 휴무" + (" + 공휴일 표시" if hol_on else ""), [d], dfm_, hol)
        # 누수 점검: 대상일의 과거전력·이동통계·기상·시간 피처는 원래와 같아야 한다
        pd.testing.assert_frame_equal(featm_.loc[on_d, CHECK_SAME], s02.feat.loc[on_d, CHECK_SAME])
        scen_rows += rws

# 추석형 연속 3일(월~수) 휴무 — 9/6~9/8. 앞선 휴무일의 전력은 실제 휴가 평일(8/2) 시간대별 실측으로 가정한다
CHUSEOK_LIKE = pd.date_range("2021-09-06", "2021-09-08", freq="D")
profile = s01.df.loc[s01.df.index.normalize() == pd.Timestamp("2021-08-02"), ["y_avg", "y_peak"]].to_numpy()
dfm3 = s01.df.copy()
for k, d in enumerate(CHUSEOK_LIKE):
    on_d = dfm3.index.normalize() == d
    dfm3.loc[on_d, ["생산량", "공장인원"]] = 0
    if k < len(CHUSEOK_LIKE) - 1:          # 다음 날의 이력으로 쓰일 날은 휴무 전력 프로파일로 바꾼다
        dfm3.loc[on_d, ["y_avg", "y_peak"]] = profile
hol3 = s02.HOLIDAYS_2021.union(CHUSEOK_LIKE)
_, rws = scenario_rows("연속 3일 휴무(추석형, 공휴일 표시)", CHUSEOK_LIKE, dfm3, hol3)
scen_rows += rws
scen_tbl = pd.DataFrame(scen_rows)
save(scen_tbl, "posthoc_whatif_extended")

# ══ 8. 근본 원인 ════════════════════════════════════════════════════════════════
print("[8] 근본 원인", flush=True)
cal = CAL.copy()
wk_shut = cal[(cal["is_shutdown"]) & (cal.index.dayofweek < 5) & (cal.index <= "2021-07-19")]
rc_rows = []
for d in wk_shut.index:
    prev_week = d - pd.Timedelta(days=7)
    rc_rows.append({"날짜": d.strftime("%Y-%m-%d"), "요일": "월화수목금"[d.dayofweek],
                    "7일 전 가동": bool(prev_week in cal.index and not cal.loc[prev_week, "is_shutdown"]),
                    "일평균전력": float(cal.loc[d, "일평균전력"]), "증강 불일치": bool(cal.loc[d, "is_aug_inconsistent"]),
                    "학습 레짐(일평균 기준)": int(np.digitize([cal.loc[d, "일평균전력"]], [30.0, REGIME_CUT])[0])})
rc_tbl = pd.DataFrame(rc_rows)
save(rc_tbl, "posthoc_rootcause_weekday_shutdowns")
# 증강 불일치일을 빼고 fold3 을 학습하면 문제의 날 라우팅이 바뀌나
Xtr3, ytr3, Xva3, yva3, idx3 = s03.get_fold_data(3, "D1")
aug_days = cal.index[cal["is_aug_inconsistent"]]
keep = ~Xtr3.index.normalize().isin(aug_days)
diag = []
for tag, (xt, yt) in (("현행 학습", (Xtr3, ytr3)), ("증강 불일치일 제외 학습", (Xtr3[keep], ytr3[keep]))):
    o = make_regime_model_v()(xt, yt, Xva3)
    rr = pd.Series(o["_route"](Xva3), index=idx3)
    for d in pd.to_datetime(["2021-07-31", "2021-08-02", "2021-08-03"]):
        w = rr[(rr.index.normalize() == d) & (rr.index.hour >= 8) & (rr.index.hour <= 17)]
        diag.append({"학습": tag, "제외된 학습 행": int((~keep).sum()) if tag != "현행 학습" else 0,
                     "날짜": d.strftime("%Y-%m-%d"), "08~17시 가동 배정": int((w == 2).sum()),
                     "그날 MAE": float(np.abs(yva3["y_avg"].to_numpy() - o["pred_avg"])[idx3.normalize() == d].mean())})
diag_tbl = pd.DataFrame(diag)
save(diag_tbl, "posthoc_rootcause_aug_excluded")

# ══ 그림 ═══════════════════════════════════════════════════════════════════════
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
COL = {"A": "#c0392b", "R": "#1f77b4", "G": "#2ca02c", "RG": "#9467bd"}
fig, ax = plt.subplots(figsize=(9, 3.2))
x = np.arange(len(down_hourly))
ax.plot(x, down_hourly["y_avg"], color="black", lw=2.2, label="실측 (실제로는 가동)")
for v in ("A", "R", "G"):
    ax.plot(x, down_hourly[f"pred[{v}]"], color=COL[v], lw=1.4, ls="-" if v != "G" else "--", label=f"{v} {VARIANTS[v][0]}")
ax.set_xticks([0, 8, 17, 24, 32, 41, 47])
ax.set_xticklabels(["7/13 0시", "8시", "17시", "7/15 0시", "8시", "17시", "23시"])
ax.set_ylabel("평균전력 (kW)")
ax.set_title("하방 위험: ERP 에 생산 0 으로 기록됐지만 실제 가동한 날 (원점에서는 계획 휴무로 보임)", fontsize=10.5)
ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "fig3_erp_missing_downside.png", dpi=150)

meta = {"tau_A": TAU_A, "total_sec": round(time.perf_counter() - T0, 1),
        "work_dir_used": True, "decision_use": "없음 (사후 진단)"}
(OUT / "posthoc_meta.json").write_bytes(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))

pd.set_option("display.width", 230)
pd.set_option("display.max_columns", 40)
for name, t in (("τ 고정", tau_tbl), ("부트스트랩", boot_tbl), ("부호표", sign_tbl), ("비휴무 MAE", harm_tbl),
                ("게이트 잡음", null_tbl), ("ERP 결측 하방", down_tbl), ("근본원인-예시", rc_tbl), ("근본원인-재학습", diag_tbl)):
    print(f"\n== {name} ==")
    print(t.round(4).to_string(index=False))
print("\n== 가정 시나리오 확장 (요약) ==")
print(scen_tbl.pivot_table(index=["시나리오", "날짜", "휴무 n일차"], columns="ID",
                           values=["예측 일평균", "08~17시 가동 배정", "경보 시간(≥τ_A)"]).round(1).to_string())
print(f"\n[완료] {meta['total_sec']}s")
