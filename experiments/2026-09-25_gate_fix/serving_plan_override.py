"""서빙 시제품 — '확정 휴무일 보정' 옵션 (수정안 R) + 안전장치. **실험 폴더의 시제품이다.**

제출 예측을 만든 평가 번들을 **그대로** 쓰고, 예측 단계에서만 보정을 건다.
대상일의 계획 생산량 합이 0(피처 is_shutdown = 1)이고 **운영자가 완전 휴무를 명시적으로 확정한 경우에만**
1단계 분류기(게이트)의 출력과 관계없이 기저 레짐(0)의 회귀모델로 평균·최대 전력을 예측한다.

안전장치 (사후 진단과 검증 워크플로 지적 반영)
- 기본값은 꺼짐(`enabled=False`). 꺼져 있으면 현행과 한 행도 다르지 않다.
- 생산량 합 0 **만으로는** 보정하지 않는다. ERP 기록 누락 가동일(7/13·7/15, 실측 136 kW)과 구분할 수 없기 때문이다.
- 공휴일표만으로도 자동 확정하지 않는다(`auto_confirm_holiday=False`). 평일 공휴일 설(2/12)은 계획 0 이었지만 실측이 47 kW 였고,
  기저 강제 시 약 24 kW 과소예측했다(학습 내 값). 확정은 **완전 휴무**를 뜻하는 운영 입력으로만 받는다.
- 플래그는 반드시 bool 이어야 한다. ERP 에서 흔한 'N'·'0'·'false' 문자열이 참으로 통과하는 것을 막는다.
- 생산 0 인데 확정이 없으면 현행 예측을 유지하고 `PLAN_ZERO_UNCONFIRMED` 경고를 낸다.
  7일 전 같은 요일이 가동(생산량 > 0 또는 일평균 전력 ≥ 70 kW)이면 `PLAN_ZERO_AFTER_OPERATING_WEEK` 도 붙인다.
- 확정인데 계획이 0 이 아니면 보정하지 않고 `SHUTDOWN_CONFIRMED_PLAN_NONZERO` 경고를 낸다.
- 보정은 평균·최대 전력(`y_avg_pred`·`y_peak_pred`, `peak_label`)에만 걸린다. 피크 분류기 확률(`peak_prob`·`peak_label_cls`)과
  분위수는 그대로다 → 보정 행이 있으면 `OVERRIDE_PARTIAL` 경고. 분류기 경보까지 끌지는 별도 정책 결정이다.
- 보정 행 수·적용 여부를 meta 에 남긴다(운영 로그용).

배포할 때는 이 함수들을 `serving/plan_override.py` 로 옮기고 상대 import 를 쓴다(여기서는 실험 폴더라 경로를 추가한다).

데모 (저장소 루트에서, FULL 실행으로 outputs/models/full/eval 이 있어야 한다):
    python -X utf8 experiments/2026-09-25_gate_fix/serving_plan_override.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                  # experiments/<실험>/ → 저장소 루트
OUT = Path(os.environ.get("KAMP_EXP_OUT") or HERE / "results").resolve()
# 제출 예측을 만든 평가 번들 (FULL 실행이 만든다. 다른 번들을 쓰려면 KAMP_EVAL_BUNDLE)
EVAL_BUNDLE = Path(os.environ.get("KAMP_EVAL_BUNDLE") or REPO / "outputs" / "models" / "full" / "eval").resolve()
sys.path.insert(0, str(REPO))

from serving import ServingInputError, load_bundle  # noqa: E402
from serving.contract import PLAN_COLUMNS, normalize_history, normalize_plan, target_of  # noqa: E402
from serving.pipeline import assemble_output, build_day_ahead_features  # noqa: E402

BASE_REGIME = 0          # 기저(30 kW 미만)
OPERATING_KW = 70.0      # 레짐 경계 · ERP 결측 판정과 같은 가동 기준


def _require_bool(name: str, v) -> None:
    if not isinstance(v, (bool, np.bool_)):
        raise ServingInputError("OVERRIDE_FLAG_INVALID", f"{name} 는 bool 이어야 한다 (받은 형식: {type(v).__name__})")


def override_planned_shutdown(parts: dict, X: pd.DataFrame, pred: dict) -> tuple[dict, bool]:
    """계획상 휴무 행 중 게이트가 기저가 아닌 레짐으로 보낸 행만 기저 레짐 회귀로 다시 예측한다.

    확정 여부(가드)는 호출자 책임이다. 반환: (보정된 예측, 기저 모델 사용 가능 여부)
    """
    cols = list(parts["columns"])
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise KeyError(f"피처 누락: {missing[:5]}")
    out = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in pred.items()}
    regs = {t: parts["regs"][t].get(BASE_REGIME, parts["regs"][t].get(str(BASE_REGIME))) for t in ("y_avg", "y_peak")}
    if any(b is None for b in regs.values()):
        return out, False
    Xf = X.reindex(columns=cols).astype(np.float64)
    fix = (Xf["is_shutdown"].to_numpy() == 1) & (np.asarray(pred["regime"]) != BASE_REGIME)
    if fix.any():
        arr = Xf.to_numpy()[fix]
        for tgt, booster in regs.items():
            out[tgt][fix] = booster.predict(arr)
        out["regime"][fix] = BASE_REGIME
    return out, True


def predict_day_plan_aware(bundle, history: pd.DataFrame, plan: pd.DataFrame, *, enabled: bool = False,
                           shutdown_confirmed: bool = False, auto_confirm_holiday: bool = False,
                           target_date=None, max_days: int | None = None):
    """(현행 출력, 서비스 출력, meta). enabled=False 이거나 확정이 없으면 두 출력은 같다."""
    for name, v in (("enabled", enabled), ("shutdown_confirmed", shutdown_confirmed),
                    ("auto_confirm_holiday", auto_confirm_holiday)):
        _require_bool(name, v)
    plan_n = normalize_plan(plan)
    if (plan_n["생산량"] < 0).any():
        raise ServingInputError("PLAN_SCHEMA", "계획 생산량이 음수다")
    target = target_of(plan_n, target_date)
    holidays = bundle.holidays_for(target)
    X, info = build_day_ahead_features(
        history, plan, columns=bundle.columns, theta=bundle.theta, holidays=holidays,
        min_days=int(bundle.history_days["min"]), max_days=max_days, target_date=target,
    )
    base = bundle.predict_features(X)
    planned_zero = bool((X["is_shutdown"] == 1).all())
    is_holiday = bool(pd.Timestamp(target).normalize() in holidays)
    confirmed = bool(shutdown_confirmed or (auto_confirm_holiday and is_holiday))
    warnings = list(info.get("warnings", []))

    if planned_zero and not confirmed:
        warnings.append("PLAN_ZERO_UNCONFIRMED: 계획 생산 0 이지만 완전 휴무 확정이 없다 — 현행 예측 유지")
        hist_n = normalize_history(history)
        wk = hist_n[hist_n["날짜"] == int((pd.Timestamp(target) - pd.Timedelta(days=7)).strftime("%Y%m%d"))]
        if wk.empty:
            warnings.append("WEEK_AGO_MISSING: 7일 전 이력이 없어 가동 여부를 확인하지 못했다")
        else:
            power = wk.loc[wk["평균"] > 0, "평균"]
            if wk["생산량"].sum() > 0 or (len(power) and power.mean() >= OPERATING_KW):
                warnings.append("PLAN_ZERO_AFTER_OPERATING_WEEK: 7일 전 같은 요일은 가동 — 운영자 확인 필요")
    if confirmed and not planned_zero:
        warnings.append("SHUTDOWN_CONFIRMED_PLAN_NONZERO: 휴무 확정인데 계획 생산량이 0 이 아니다 — 보정하지 않음")

    apply = bool(enabled and planned_zero and confirmed)
    served = base
    if apply:
        served, ok = override_planned_shutdown(bundle.parts, X, base)
        if not ok:
            warnings.append("OVERRIDE_UNAVAILABLE: 번들에 기저 레짐 모델이 없어 보정하지 않았다")
            apply = False
    n_fix = int((np.asarray(served["regime"]) != np.asarray(base["regime"])).sum())
    if n_fix:
        warnings.append("OVERRIDE_PARTIAL: peak_prob·peak_label_cls·분위수는 보정되지 않았다")

    kw = dict(tau=bundle.tau, tau_cls=bundle.tau_cls, calibrator=bundle.calibrator, halfwidths=bundle.halfwidths)
    meta = {"bundle_id": bundle.bundle_id, "role": bundle.role, **info, "target_date": str(pd.Timestamp(target).date()),
            "planned_zero": planned_zero, "holiday": is_holiday, "confirmed": confirmed,
            "override_applied": apply, "rows_overridden": n_fix, "warnings": warnings}
    return assemble_output(X, base, **kw), assemble_output(X, served, **kw), meta


def main() -> int:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    bundle = load_bundle(EVAL_BUNDLE)
    raw = pd.read_csv(REPO / "data" / "okm_augumented_2021.csv", encoding="utf-8-sig")
    ymd = raw["날짜"].astype(int)

    def day_inputs(d: pd.Timestamp, zero: bool):
        hist = raw[ymd <= int((d - pd.Timedelta(days=1)).strftime("%Y%m%d"))]   # 전날까지 이력 전체
        plan = raw.loc[ymd == int(d.strftime("%Y%m%d")), PLAN_COLUMNS].copy()
        if zero:
            plan["생산량"] = 0          # 가정: 이날을 계획 휴무로 바꾼다
            plan["공장인원"] = 0
        return hist, plan

    rows = []
    for d in pd.date_range("2021-09-01", "2021-09-14", freq="D"):
        if d.dayofweek >= 5:
            continue
        hist, plan = day_inputs(d, zero=True)
        cur, srv_c, meta_c = predict_day_plan_aware(bundle, hist, plan, enabled=True, shutdown_confirmed=True)
        _, srv_u, meta_u = predict_day_plan_aware(bundle, hist, plan, enabled=True, shutdown_confirmed=False)
        _, srv_off, _ = predict_day_plan_aware(bundle, hist, plan, enabled=False, shutdown_confirmed=True)
        assert srv_off.equals(cur), "옵션이 꺼져 있으면 현행과 같아야 한다"
        assert srv_u.equals(cur), "확정이 없으면 보정하지 않아야 한다"
        rows.append({"날짜": meta_c["target_date"], "보정된 행(확정 시)": meta_c["rows_overridden"],
                     "현행 예측 일평균": float(cur["y_avg_pred"].mean()),
                     "보정 예측 일평균(확정 시)": float(srv_c["y_avg_pred"].mean()),
                     "현행 peak_label 시간": int(cur["peak_label"].sum()),
                     "보정 peak_label 시간(확정 시)": int(srv_c["peak_label"].sum()),
                     "peak_label_cls 시간(보정 안 됨)": int(srv_c["peak_label_cls"].sum()),
                     "미확정 시 경고": " / ".join(w.split(":")[0] for w in meta_u["warnings"])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "serving_whatif.csv", index=False, encoding="utf-8-sig")
    log(res.round(3).to_string(index=False))

    # 실험 스크립트의 같은 시나리오(A=현행, R=보정)와 날짜로 맞춰 교차 확인 — 구현 일치 점검이지 효과 검증이 아니다
    exp = pd.read_csv(OUT / "gate_fix_whatif.csv", encoding="utf-8-sig")
    m = res.merge(exp, on="날짜", validate="1:1")
    assert len(m) == len(res) == len(exp)
    da = float(np.abs(m["현행 예측 일평균"] - m["A 예측 일평균"]).max())
    dr = float(np.abs(m["보정 예측 일평균(확정 시)"] - m["R 예측 일평균"]).max())
    log(f"\n구현 일치 확인(일평균 {len(m)}개): 현행 최대 차이 {da:.2e} kW · 보정 최대 차이 {dr:.2e} kW")
    ok = da < 1e-6 and dr < 1e-6

    # 추가 점검 — 실제 계획일은 켜고 확정해도 불변, 문자열 플래그는 거부, 공휴일은 자동 확정하지 않음
    d = pd.Timestamp("2021-09-07")
    hist, plan = day_inputs(d, zero=False)
    cur, srv, meta = predict_day_plan_aware(bundle, hist, plan, enabled=True, shutdown_confirmed=True)
    assert srv.equals(cur) and "SHUTDOWN_CONFIRMED_PLAN_NONZERO" in " ".join(meta["warnings"])
    try:
        predict_day_plan_aware(bundle, hist, plan, enabled=True, shutdown_confirmed="N")
        raise AssertionError("문자열 플래그가 통과했다")
    except ServingInputError as e:
        assert e.code == "OVERRIDE_FLAG_INVALID"
    hol = pd.Timestamp("2021-08-16")          # 공휴일표에 있는 날(대체공휴일) — 실제로는 가동했다
    hist, plan = day_inputs(hol, zero=True)
    cur, srv, meta = predict_day_plan_aware(bundle, hist, plan, enabled=True, shutdown_confirmed=False)
    assert meta["holiday"] and not meta["override_applied"] and srv.equals(cur)
    log("추가 점검 통과: 실제 계획일 불변(경고) · 문자열 플래그 거부 · 공휴일 자동 확정 안 함")
    log("교차 확인: " + ("일치" if ok else "불일치"))
    (OUT / "serving_demo.log").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
