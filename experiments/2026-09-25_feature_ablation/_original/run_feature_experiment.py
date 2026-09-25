"""피처 제거 확인 실험 — 계획은 ../PLAN.md.

최종모델(2단계 레짐 3분류)을 FULL 설정 그대로 fold 2~5 에서 돌려
피처군·개별 피처 제거, 휴무 인지형 지연변수 교체의 효과를 fold·구간별로 잰다.
테스트 구간은 읽지 않는다.

실행 (KAMP 피처실험 폴더에서):
    PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiment/run_feature_experiment.py
"""
from __future__ import annotations

import ast
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "KAMP-Competition"
OUT = HERE / "results"

if os.environ.get("PYTHONHASHSEED") != "42" or os.environ.get("KAMP_FAST", "0") != "0":
    sys.exit("PYTHONHASHSEED=42, KAMP_FAST=0 으로 실행해야 한다")

OUT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)  # 원본 코드는 data/·outputs/ 를 상대경로로 찾는다
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# 재현 기준값 — s00~s03 이 쓰지 않는 표지만, import 전에 읽어 둔다
FINAL = "2단계 레짐(3분류)"
EXP_SCORE = pd.read_csv("outputs/tables/ch2_scorecard.csv", encoding="utf-8-sig").set_index("모델").loc[FINAL]
EXP_ABL = pd.read_csv("outputs/tables/ch2_ablation.csv", encoding="utf-8-sig").set_index("제거 피처군")

T0 = time.perf_counter()
import s00_env as s00  # noqa: E402
import s01_diagnose as s01  # noqa: E402
import s02_features as s02  # noqa: E402
import s03_split as s03  # noqa: E402
import lightgbm as lgb  # noqa: E402

print(f"[준비] s00~s03 실행 {time.perf_counter() - T0:.1f}s", flush=True)
assert s00.FAST is False, "FULL 설정이어야 한다"

# ── s05 에서 레짐 모델·CV 루프 원문만 추출해 실행 (s05 import 는 5장 전체를 돈다) ──
WANT = {"LGB_BASE", "N_TRIALS", "N_ESTIMATORS", "_TIMING", "run_cv", "REGIME_CUT", "make_regime_model"}
ns: dict = {}
for _m in (s00, s01, s02, s03):
    ns.update(vars(_m))
ns.update(time=time, lgb=lgb)
_nodes = []
for node in ast.parse((REPO / "src" / "s05_models.py").read_text(encoding="utf-8")).body:
    if isinstance(node, ast.FunctionDef):
        names = {node.name}
    elif isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = {node.target.id}
    else:
        continue
    if names & WANT:
        _nodes.append(node)
exec(compile(ast.Module(body=_nodes, type_ignores=[]), "s05_models.py", "exec"), ns)
missing = WANT - set(ns)
assert not missing, f"s05 추출 누락: {missing}"
assert ns["N_ESTIMATORS"] == 800, ns["N_ESTIMATORS"]
run_cv, make_regime_model = ns["run_cv"], ns["make_regime_model"]

THETA = s01.THETA
CAL = s01.operating_calendar
FEATURE_GROUPS = s02.FEATURE_GROUPS
FEATURE_COLS = list(s02.FEATURE_COLS)
LABELS = s02.FEATURE_LABELS
ACTIVE_FOLDS = s03.ACTIVE_FOLDS
ORIG_FEAT = s03.feat
regression_metrics = s03.regression_metrics
classification_metrics = s03.classification_metrics
paired_bootstrap_mae = s03.paired_bootstrap_mae
NORMAL_FOLDS = (2, 5)  # 하계휴가·재가동이 없는 fold


# ── 구간 정의 (일 단위) ────────────────────────────────────────────────
def day_flags(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """대상일과 참조일의 휴무 여부가 다른 날 = 지연변수가 엉뚱한 상태를 가리키는 날."""
    day = index.normalize()
    st = CAL["is_shutdown"].astype(int)
    s0 = st.reindex(day).to_numpy()
    s1 = st.reindex(day - pd.Timedelta(days=1)).to_numpy()
    s7 = st.reindex(day - pd.Timedelta(days=7)).to_numpy()
    return {
        "불일치168": s0 != s7,
        "일치168": s0 == s7,
        "불일치24": s0 != s1,
        "휴무일": s0 == 1,
        "가동일": s0 == 0,
    }


# ── 휴무 인지형 지연변수 ───────────────────────────────────────────────
def status_matched_lag(base: pd.DataFrame, col: str, lag_h: int, step_days: int, max_steps: int):
    """참조일의 휴무 여부가 대상일과 다르면 step_days 씩 물러나 같은 상태의 날 같은 시각 값을 쓴다.

    참조일은 항상 대상일 − lag_h 이전이라 원점(대상일 00:00) 기준 인과적이다.
    """
    status = CAL["is_shutdown"].astype(int)
    day = base.index.normalize()
    hour_td = pd.to_timedelta(base.index.hour, unit="h")
    tgt = status.reindex(day).fillna(-1).to_numpy()
    y = base[col]
    out = np.full(len(base), np.nan)
    walked = np.full(len(base), -1)
    todo = np.ones(len(base), bool)
    ref = day - pd.Timedelta(hours=lag_h)
    for step in range(max_steps + 1):
        st = status.reindex(ref).fillna(-2).to_numpy()
        hit = todo & (st == tgt)
        vals = y.reindex(ref + hour_td).to_numpy()
        out[hit] = vals[hit]
        walked[hit] = step
        todo &= ~hit
        if not todo.any():
            break
        ref = ref - pd.Timedelta(days=step_days)
    return out, walked


def make_sa_feat(lags) -> tuple[pd.DataFrame, list[dict]]:
    f = ORIG_FEAT.copy()
    usable = s03.usable_mask(f, "D1").to_numpy()
    info = []
    for col in ("y_avg", "y_peak"):
        for lag in lags:
            step, mx = (7, 5) if lag == 168 else (1, 14)
            vals, walked = status_matched_lag(f, col, lag, step, mx)
            name = f"{col}_lag{lag}"
            raw = f[name].to_numpy()
            same = walked == 0
            # 물러나지 않은 행은 원래 lag 와 정확히 같아야 한다
            assert np.array_equal(vals[same], raw[same], equal_nan=True), name
            f[name] = vals
            info.append({
                "피처": name, "사용가능행": int(usable.sum()),
                "교체된 행": int(((walked > 0) & usable).sum()),
                "결측 행": int((np.isnan(vals) & usable).sum()),
                "원래 결측 행": int((np.isnan(raw) & usable).sum()),
            })
    return f, info


# ── 변형 실행·요약 ─────────────────────────────────────────────────────
RESULTS: dict[tuple[str, str], dict] = {}


def run_variant(vid: str, desc: str, *, features=None, gate_features=None, feat_df=None, cond="D1") -> dict:
    s03.feat = ORIG_FEAT if feat_df is None else feat_df
    try:
        t = time.perf_counter()
        res = run_cv(vid, make_regime_model(3, gate_features=gate_features), cond, features=features)
        sec = time.perf_counter() - t
    finally:
        s03.feat = ORIG_FEAT
    res.update(vid=vid, desc=desc, n_feat=len(features) if features is not None else len(FEATURE_COLS), sec=sec)
    RESULTS[(cond, vid)] = res
    oof = res["oof"]
    print(f"  [{cond}] {vid:<22s} MAE {s03.mae(oof['y_avg'], oof['pred_avg']):7.4f}  ({sec:4.1f}s)", flush=True)
    return res


def summarize(res: dict, base: dict | None) -> dict:
    oof = res["oof"]
    y, p = oof["y_avg"].to_numpy(), oof["pred_avg"].to_numpy()
    reg = regression_metrics(oof["y_avg"], oof["pred_avg"], THETA)
    cls = classification_metrics(oof["y_cls"], (oof["pred_peak"] >= res["tau"]).astype(int), oof["pred_peak"])
    fm = res["fold_metrics"].set_index("fold")["MAE"]
    ae = np.abs(y - p)
    flags = day_flags(oof.index)
    row = {
        "ID": res["vid"], "설명": res["desc"], "조건": res["cond"], "피처수": res["n_feat"],
        "MAE": reg["MAE"], "Peak-MAE": reg["Peak-MAE"], "Recall": cls["Recall"], "F1": cls["F1"],
        "FP": cls["FP"], "FN": cls["FN"], "τ": res["tau"], "fold표준편차": float(fm.std()),
    }
    for f in ACTIVE_FOLDS:
        row[f"fold{f}"] = float(fm[f])
    for k, m in flags.items():
        row[f"MAE_{k}"] = float(ae[m].mean()) if m.any() else np.nan
    if base is None:
        return row
    b = base["oof"]
    assert b.index.equals(oof.index), "변형 간 OOF 행이 달라 짝지은 비교가 불가능하다"
    bp = b["pred_avg"].to_numpy()
    brow = summarize(base, None)
    for k in ("MAE", "Peak-MAE", "Recall", "F1", "fold표준편차", *[f"fold{f}" for f in ACTIVE_FOLDS],
              *[f"MAE_{k}" for k in flags]):
        row[f"Δ{k}"] = row[k] - brow[k]
    # paired_bootstrap_mae 의 diff = MAE(A) − MAE(V) → 부호를 뒤집어 V − A 로 적는다
    bs = paired_bootstrap_mae(y, bp, p, index=oof.index)
    row["ΔMAE_CI_low"], row["ΔMAE_CI_high"], row["p"] = -bs["ci_high"], -bs["ci_low"], bs["p_value"]
    m = flags["일치168"]
    bs2 = paired_bootstrap_mae(y[m], bp[m], p[m], index=oof.index[m])
    row["Δ일치168_CI_low"], row["Δ일치168_CI_high"] = -bs2["ci_high"], -bs2["ci_low"]
    folds_better = sum(row[f"Δfold{f}"] < 0 for f in ACTIVE_FOLDS)
    row["좋아진 fold 수"] = int(folds_better)
    return row


def table(cond: str, ids) -> pd.DataFrame:
    base = RESULTS[(cond, "A")]
    rows = [summarize(RESULTS[(cond, v)], None if v == "A" else base) for v in ids]
    return pd.DataFrame(rows)


def save(df: pd.DataFrame, name: str):
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")


# ══════════════════════════════════════════════════════════════════════
print(f"[설정] 피처 {len(FEATURE_COLS)}개, fold {ACTIVE_FOLDS}, 트리 {ns['N_ESTIMATORS']}", flush=True)

# 1) 기준 + 피처군 제거 (기존 Ablation 재현)
print("[1] 기준·피처군 제거", flush=True)
run_variant("A", "기준(44개)")
GROUP_IDS = []
for g, cols in FEATURE_GROUPS.items():
    keep = [c for c in FEATURE_COLS if c not in set(cols)]
    gate = keep if g == "생산정보" else None  # 원본 Ablation 과 같이 게이트도 생산량 비의존으로
    vid = f"G-{g}"
    run_variant(vid, f"{g} {len(cols)}개 제거" + (" (게이트도 교체)" if gate else ""), features=keep, gate_features=gate)
    GROUP_IDS.append(vid)

# 2) 재현 확인 — 어긋나면 멈춘다
a = summarize(RESULTS[("D1", "A")], None)
checks = [
    ("기준 OOF MAE", a["MAE"], float(EXP_SCORE["1) OOF MAE"]), 1e-4),
    ("기준 Peak-MAE", a["Peak-MAE"], float(EXP_SCORE["1) OOF Peak-MAE"]), 1e-4),
    ("기준 Recall", a["Recall"], float(EXP_SCORE["2) OOF Recall"]), 1e-4),
    ("기준 fold 표준편차", a["fold표준편차"], float(EXP_SCORE["3) fold MAE 표준편차"]), 1e-4),
]
ABL_MAP = {"과거전력": "과거 전력 지연변수", "생산정보": "생산량·공장인원", "공정상태": "공정상태 변수", "기상정보": "기상정보"}
for g, label in ABL_MAP.items():
    r = summarize(RESULTS[("D1", f"G-{g}")], None)
    checks.append((f"Ablation {label} MAE", r["MAE"], float(EXP_ABL.loc[label, "MAE"]), 6e-4))
    checks.append((f"Ablation {label} Recall", r["Recall"], float(EXP_ABL.loc[label, "Recall"]), 1e-4))
repro = pd.DataFrame(
    [{"항목": n, "이번 실행": round(v, 6), "커밋된 FULL": e, "일치": abs(v - e) <= tol} for n, v, e, tol in checks]
)
save(repro, "repro_check")
print(repro.to_string(index=False), flush=True)
if not repro["일치"].all():
    sys.exit("[중단] 커밋된 FULL 결과를 재현하지 못했다 — repro_check.csv 확인")

# 3) 지연변수 세분·조합·휴무 인지형
print("[3] 지연 세분·조합·휴무 인지형", flush=True)
LAG_IDS = []
for lag in (24, 48, 168):
    drop = {f"y_avg_lag{lag}", f"y_peak_lag{lag}"}
    run_variant(f"L{lag}", f"{lag}시간 지연 2개 제거", features=[c for c in FEATURE_COLS if c not in drop])
    LAG_IDS.append(f"L{lag}")
drop_bw = set(FEATURE_GROUPS["과거전력"]) | set(FEATURE_GROUPS["기상정보"])
run_variant("B+W", "지연 6개 + 기상 6개 제거", features=[c for c in FEATURE_COLS if c not in drop_bw])
SA_FEAT_ALL, sa_info_all = make_sa_feat((24, 48, 168))
SA_FEAT_168, sa_info_168 = make_sa_feat((168,))
save(pd.DataFrame([{"변형": "SA-all", **r} for r in sa_info_all] + [{"변형": "SA-168", **r} for r in sa_info_168]),
     "sa_coverage")
run_variant("SA-all", "지연 6개를 휴무 인지형으로 교체", feat_df=SA_FEAT_ALL)
run_variant("SA-168", "lag168 2개만 휴무 인지형으로 교체", feat_df=SA_FEAT_168)
EXTRA_IDS = [*LAG_IDS, "B+W", "SA-all", "SA-168"]

# 4) 잡음 기준선
print("[4] 잡음 기준선 (순수 잡음 피처 1개 추가)", flush=True)
NULL_IDS = []
for k in range(5):
    f = ORIG_FEAT.copy()
    f["noise"] = np.random.default_rng(1000 + k).standard_normal(len(f))
    run_variant(f"NULL-{k}", f"잡음 피처 추가(시드 {1000 + k})", features=[*FEATURE_COLS, "noise"], feat_df=f)
    NULL_IDS.append(f"NULL-{k}")

# 5) 개별 피처 하나씩 제거
print("[5] LOFO 44회", flush=True)
LOFO_IDS = []
for c in FEATURE_COLS:
    vid = f"LOFO-{c}"
    run_variant(vid, f"{LABELS.get(c, c)} 제거", features=[x for x in FEATURE_COLS if x != c])
    LOFO_IDS.append(vid)

# ── D1 요약·판정 ───────────────────────────────────────────────────────
d1 = table("D1", ["A", *GROUP_IDS, *EXTRA_IDS, *NULL_IDS, *LOFO_IDS])
null = d1[d1["ID"].isin(NULL_IDS)]
NOISE_MAE = float(null["ΔMAE"].abs().max())
NOISE_PEAK = float(null["ΔPeak-MAE"].abs().max())
print(f"[판정] 잡음 폭 MAE {NOISE_MAE:.4f} / Peak-MAE {NOISE_PEAK:.4f}", flush=True)

cand = d1[~d1["ID"].isin(["A", *NULL_IDS])].copy()
cand["c1 CI가 0 미포함(개선)"] = cand["ΔMAE_CI_high"] < 0
cand["c2 정상 fold 2·5 악화 ≤ 잡음"] = (cand["Δfold2"] <= NOISE_MAE) & (cand["Δfold5"] <= NOISE_MAE)
cand["c3 일치일 악화 ≤ 잡음"] = cand["ΔMAE_일치168"] <= NOISE_MAE
cand["c4 Recall·Peak-MAE"] = (cand["ΔRecall"] >= -0.01) & (cand["ΔPeak-MAE"] <= NOISE_PEAK)
cand["c1~c4 통과"] = cand[[c for c in cand.columns if c.startswith("c") and c[1].isdigit()]].all(axis=1)
cand["LOFO 표시(잡음 초과 개선·3/4 fold)"] = (
    cand["ID"].str.startswith("LOFO-") & (cand["ΔMAE"] < -NOISE_MAE) & (cand["좋아진 fold 수"] >= 3)
)
save(d1, "summary_d1")
save(cand, "criteria_d1")

# 6) D2 강건성 — 고정 목록 + c1~c4 통과 + LOFO 표시
print("[6] D2 강건성", flush=True)
fixed = ["G-과거전력", "L168", "B+W", "SA-all", "SA-168"]
extra = cand.loc[cand["c1~c4 통과"] | cand["LOFO 표시(잡음 초과 개선·3/4 fold)"], "ID"].tolist()
d2_ids = list(dict.fromkeys([*fixed, *extra]))
feat_map = {"SA-all": SA_FEAT_ALL, "SA-168": SA_FEAT_168}
run_variant("A", "기준(44개)", cond="D2")
for vid in d2_ids:
    src = RESULTS[("D1", vid)]
    # D1 에서 쓴 피처 목록을 다시 만든다
    if vid.startswith("G-"):
        g = vid[2:]
        feats = [c for c in FEATURE_COLS if c not in set(FEATURE_GROUPS[g])]
        run_variant(vid, src["desc"], features=feats, gate_features=feats if g == "생산정보" else None, cond="D2")
    elif vid.startswith("L") and vid[1:].isdigit():
        lag = int(vid[1:])
        drop = {f"y_avg_lag{lag}", f"y_peak_lag{lag}"}
        run_variant(vid, src["desc"], features=[c for c in FEATURE_COLS if c not in drop], cond="D2")
    elif vid == "B+W":
        run_variant(vid, src["desc"], features=[c for c in FEATURE_COLS if c not in drop_bw], cond="D2")
    elif vid in feat_map:
        run_variant(vid, src["desc"], feat_df=feat_map[vid], cond="D2")
    elif vid.startswith("LOFO-"):
        c = vid[5:]
        run_variant(vid, src["desc"], features=[x for x in FEATURE_COLS if x != c], cond="D2")
d2 = table("D2", ["A", *d2_ids])
save(d2, "summary_d2")

# ── OOF 예측 보관 (D1) ─────────────────────────────────────────────────
base_oof = RESULTS[("D1", "A")]["oof"]
wide = base_oof[["fold", "y_avg", "y_peak", "y_cls"]].copy()
for (cond, vid), res in RESULTS.items():
    if cond == "D1":
        wide[f"pred_avg[{vid}]"] = res["oof"]["pred_avg"].to_numpy()
wide.index.name = "ts"
wide.reset_index().to_csv(OUT / "oof_d1.csv", index=False, encoding="utf-8-sig")

# 일별 오차 (기준 vs 지연 제거 vs 휴무 인지형) — 어느 날이 차이를 만드는지
day = base_oof.index.normalize()
flags = day_flags(base_oof.index)
daily = pd.DataFrame({
    "날짜": day, "fold": base_oof["fold"].to_numpy(),
    "휴무일": flags["휴무일"], "불일치168": flags["불일치168"], "불일치24": flags["불일치24"],
    **{f"AE[{v}]": np.abs(base_oof["y_avg"].to_numpy() - RESULTS[("D1", v)]["oof"]["pred_avg"].to_numpy())
       for v in ("A", "G-과거전력", "L168", "SA-all", "SA-168")},
}).groupby("날짜").agg({"fold": "first", "휴무일": "first", "불일치168": "first", "불일치24": "first",
                       **{f"AE[{v}]": "mean" for v in ("A", "G-과거전력", "L168", "SA-all", "SA-168")}})
daily.reset_index().to_csv(OUT / "daily_error_d1.csv", index=False, encoding="utf-8-sig")

meta = {
    "noise_band_mae": NOISE_MAE, "noise_band_peak_mae": NOISE_PEAK,
    "n_variants_d1": int(len(d1)), "n_variants_d2": int(len(d2)),
    "d2_ids": d2_ids, "total_sec": round(time.perf_counter() - T0, 1),
    "python": platform.python_version(), "lightgbm": lgb.__version__,
    "pandas": pd.__version__, "numpy": np.__version__,
}
(OUT / "meta.json").write_bytes(json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
print(f"[완료] {meta['total_sec']}s — 결과: experiment/results/", flush=True)
