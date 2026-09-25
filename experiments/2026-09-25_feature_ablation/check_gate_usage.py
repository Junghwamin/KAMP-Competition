"""사후 점검: 게이트가 휴무 피처를 실제로 쓰는지, 08-02·03 을 어느 레짐으로 보냈는지 확인한다.

실행 (저장소 루트에서): PYTHONHASHSEED=42 KAMP_FAST=0 python -X utf8 experiments/2026-09-25_feature_ablation/check_gate_usage.py
"""
from __future__ import annotations

import ast
import contextlib
import io
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
(WORK / "data").mkdir(parents=True, exist_ok=True)
if not (WORK / "data" / "okm_augumented_2021.csv").exists():
    shutil.copy2(REPO / "data" / "okm_augumented_2021.csv", WORK / "data" / "okm_augumented_2021.csv")
os.chdir(WORK)
sys.path.insert(0, str(REPO / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

with contextlib.redirect_stdout(io.StringIO()):
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
nodes = []
for node in ast.parse((REPO / "src" / "s05_models.py").read_text(encoding="utf-8")).body:
    names = set()
    if isinstance(node, ast.FunctionDef):
        names = {node.name}
    elif isinstance(node, ast.Assign):
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        names = {node.target.id}
    if names & WANT:
        nodes.append(node)
exec(compile(ast.Module(body=nodes, type_ignores=[]), "s05_models.py", "exec"), ns)
assert ns["N_ESTIMATORS"] == 800

f = s02.feat
print("shutdown_nth>0 ⇔ is_shutdown==1 (전 행):", bool(((f["shutdown_nth"] > 0) == (f["is_shutdown"] == 1)).all()))
print("dow==6 ⇔ is_sun==1 (전 행):", bool(((f["dow"] == 6) == (f["is_sun"] == 1)).all()))

cols = ["is_shutdown", "shutdown_nth", "is_sun", "prod", "prod_cum_day", "y_avg_lag24", "y_avg_lag48", "y_avg_lag168", "is_holiday"]
for fold in (3, 4):
    Xtr, ytr, Xva, yva, idx = s03.get_fold_data(fold, "D1")
    out = ns["make_regime_model"](3)(Xtr, ytr, Xva)
    art = out["_artifacts"]
    gate = art["gate"]
    imp = pd.Series(gate.booster_.feature_importance("split"), index=gate.booster_.feature_name())
    print(f"\n[fold{fold}] 게이트 분기 횟수:", imp.reindex(cols).astype(int).to_dict())
    rr = gate.predict(Xva[art["gcols"]])
    day = pd.Series(rr, index=idx)
    for d in ("2021-07-31", "2021-08-02", "2021-08-03", "2021-08-09"):
        if d in day.index.strftime("%Y-%m-%d"):
            sel = day.loc[d]
            print(f"  {d} 08~17시 게이트 레짐:", sel.between_time("08:00", "17:00").value_counts().sort_index().to_dict(),
                  " 실측 y_avg 평균:", round(float(yva.loc[d, "y_avg"].mean()), 1))
