"""제안서·결과 문서용 그림 (결과 CSV 만 읽는다 — 파이프라인 재실행 없음).

    python -X utf8 experiments/2026-09-25_gate_fix/make_figures.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("KAMP_EXP_OUT") or HERE / "results").resolve()
FIG = Path(os.environ.get("KAMP_EXP_FIG") or HERE / "figures").resolve()
FIG.mkdir(parents=True, exist_ok=True)
THETA = 187.0
TAU_A = float(json.loads((RES / "posthoc_meta.json").read_bytes())["tau_A"])
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
C_A, C_R, C_G = "#c0392b", "#1f77b4", "#2ca02c"

# ── 그림 1: 문제의 이틀 (발견 데이터) ───────────────────────────────────────
hr = pd.read_csv(RES / "gate_fix_two_days_hourly.csv", encoding="utf-8-sig")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), sharey=True)
for ax, (label, cols) in zip(axes, (("현행 A", [("pred[A]", C_A, "-")]),
                                    ("수정안 R · G", [("pred[R]", C_R, "-"), ("pred[G]", C_G, "--")]))):
    x = np.arange(len(hr))
    ax.plot(x, hr["y_avg"], color="black", lw=2.4, label="실측")
    for c, col, ls in cols:
        ax.plot(x, hr[c], color=col, lw=1.5, ls=ls, label=c[5:-1] + (" 계획 휴무 강제" if "R" in c else " 게이트 입력 제한" if "G" in c else " 현행"))
    ax.set_xticks([0, 8, 17, 24, 32, 41, 47])
    ax.set_xticklabels(["8/2 0시", "8시", "17시", "8/3 0시", "8시", "17시", "23시"], fontsize=8)
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
axes[0].set_ylabel("평균전력 (kW)")
fig.suptitle("하계휴가 중 평일(계획 생산 0) — 교차검증 예측 · 문제를 발견한 바로 그 이틀", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig1_two_days.png", dpi=150)
plt.close(fig)

# ── 그림 2: 가정 시나리오 (정답 없음) ───────────────────────────────────────
sc = pd.read_csv(RES / "posthoc_whatif_extended.csv", encoding="utf-8-sig")
single = sc[sc["시나리오"] == "단일 평일 휴무"].pivot(index="날짜", columns="ID")
three = sc[sc["시나리오"].str.startswith("연속 3일")].pivot(index="날짜", columns="ID")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [10, 3.4]})
for ax, t, title in ((axes[0], single, "단일 평일을 계획 휴무로 가정 (10일)"),
                     (axes[1], three, "추석형 연속 3일(월~수)\n앞선 휴무일 전력은 8/2 프로필로 근사")):
    xs = np.arange(len(t))
    ax.bar(xs - 0.2, t[("예측 일평균", "A")], width=0.4, color=C_A, label="현행 A")
    ax.bar(xs + 0.2, t[("예측 일평균", "R")], width=0.4, color=C_R, label="수정안 R")
    for i, h in enumerate(t[("경보 시간(≥τ_A)", "A")]):
        if h:
            ax.text(xs[i] - 0.2, t[("예측 일평균", "A")].iloc[i] + 2, f"경보\n{h}h", ha="center", fontsize=7, color=C_A)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d[5:]}" for d in t.index], fontsize=8)
    ax.set_title(title, fontsize=9.5)
    ax.set_ylim(0, 160)
    ax.grid(alpha=0.3, axis="y")
axes[0].set_ylabel("예측 일평균 (kW)")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("가정 시나리오 — 정답(실측) 없음. 이 날들은 실제로는 가동일(103~139 kW)이다. "
             "현행 게이트가 '계획 0 + 지난주 가동'을 가동으로 보는 기전만 보여 준다", fontsize=9.5)
fig.tight_layout()
fig.savefig(FIG / "fig2_whatif_weekday_shutdown.png", dpi=150)
plt.close(fig)

# ── 그림 3: ERP 결측 가동일 (하방 위험) ─────────────────────────────────────
hp = pd.read_csv(RES / "posthoc_erp_missing_hourly_peaks.csv", encoding="utf-8-sig", parse_dates=["ts"])
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), sharey=True)
for ax, day in zip(axes, ("2021-07-13", "2021-07-15")):
    a = hp[(hp["방식"] == "A") & (hp["ts"].dt.strftime("%Y-%m-%d") == day)].reset_index(drop=True)
    r = hp[(hp["방식"] == "R") & (hp["ts"].dt.strftime("%Y-%m-%d") == day)].reset_index(drop=True)
    h = np.arange(24)
    ax.plot(h, a["y_avg"], color="black", lw=2.4, label="실측 평균전력")
    ax.plot(h, a["pred_avg"], color=C_A, lw=1.5, label="현행 A 예측")
    ax.plot(h, r["pred_avg"], color=C_R, lw=1.5, label="수정안 R 예측")
    pk = a["y_peak"] >= THETA
    al = a["pred_peak"] >= TAU_A
    ax.scatter(h[pk], np.full(pk.sum(), 205), marker="v", color="black", s=28, label="실제 피크(≥187)")
    ax.scatter(h[al], np.full(al.sum(), 215), marker="|", color=C_A, s=80, label="현행 경보(≥τ)")
    ax.set_title(f"{day[5:].replace('-', '/')} — 현행 적중 {int((pk & al).sum())}/{int(pk.sum())}, 오경보 {int((al & ~pk).sum())}",
                 fontsize=9.5)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xlabel("시")
    ax.set_ylim(0, 225)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("kW")
axes[1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("하방 위험: ERP 에 생산 0 으로 기록됐지만 실제 가동한 날 (fold2 학습창 모델 · 원점에서는 계획 휴무로 보임)",
             fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig3_erp_missing_downside.png", dpi=150)
plt.close(fig)
print("그림 3장 저장")
