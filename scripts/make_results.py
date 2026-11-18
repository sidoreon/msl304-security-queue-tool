"""Reproduce every number and figure used in the report and slides.

    python scripts/make_results.py      ->  report_assets/*.png, report_assets/results.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.queueing import mmc_metrics, mm1_split_metrics, pooling_benefit, lanes_needed
from src.priority import fast_lane_metrics, cobham_metrics
from src.staffing import (Params, DESIGNS, DESIGN_LABELS, compare_designs, optimize_staffing,
                          optimize_staffing_mip, baseline_plan, saving, cost_curve, evaluate_block)
from src.breakeven import breakeven_table, net_benefit_curve
from src.simulation import simulate_block, passes, StressOptions
from src.sample_data import sample_profile
from src import sensitivity as sens

OUT = os.path.join(ROOT, "report_assets")
os.makedirs(OUT, exist_ok=True)
# Palette: ivory background, US Open blue, clay-court orange.
IVORY, INK, GRID = "#FBF8EF", "#1E2633", "#E6E0D0"
BLUE, BLUE_LT = "#1D4F91", "#6E9BD1"
CLAY, CLAY_LT = "#C65A2E", "#E3A47E"
C = {"pooled": BLUE, "separate": CLAY, "fast_lane": BLUE_LT, "inlane_priority": CLAY_LT,
     "ink": INK, "muted": "#8C8577", "red": CLAY, "violet": BLUE_LT}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.facecolor": IVORY, "figure.facecolor": IVORY, "savefig.facecolor": IVORY,
    "axes.edgecolor": "#B8B09D", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 180, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
    "legend.frameon": False, "lines.solid_capstyle": "round"})

R: dict = {}
P = Params()
prof_df = sample_profile()
prof = prof_df["lam"].values
labels = list(prof_df["start"])
K = len(prof)
peak = int(np.argmax(prof))
R["profile"] = {"K": K, "peak_block": peak + 1, "peak_time": labels[peak],
                "peak_lam": float(prof[peak]), "min_lam": float(prof.min()),
                "pax_day": float(prof.sum() * P.delta), "lams": prof.tolist(), "labels": labels}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def xt(ax):
    ax.set_xlabel("time of day (start of 30-minute block)")
    ax.set_xticks(range(0, K, 4))
    ax.set_xticklabels(labels[::4], rotation=0)


# ------------------------------------------------------------------ 1 profile
fig, ax = plt.subplots(figsize=(7.5, 2.8))
ax.plot(range(K), prof, color=C["pooled"], lw=2, marker="o", ms=3)
ax.set_ylabel("λ_t (passengers/min)")
ax.set_title("Assumed arrival profile, 30-minute blocks (05:00 to 23:00)")
xt(ax)
save(fig, "fig_profile.png")

# ------------------------------------------------------------------ 2 pooling
R["hand_check"] = pooling_benefit(1, 1, 2)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
cc = 4
lams = np.linspace(0.05, cc * P.mu * 0.97, 200)
axs[0].plot(lams, [mm1_split_metrics(l, P.mu, cc).Wq for l in lams], color=C["separate"], lw=2,
            label="Separate (4 × M/M/1)")
axs[0].plot(lams, [mmc_metrics(l, P.mu, cc).Wq for l in lams], color=C["pooled"], lw=2,
            label="Pooled (M/M/4)")
axs[0].set_ylim(0, 6)
axs[0].set_xlabel("arrival rate λ (passengers/min)")
axs[0].set_ylabel("mean wait W_q (min)")
axs[0].set_title("Mean wait, c = 4, μ = 2")
axs[0].legend()
rhos = np.linspace(0.05, 0.97, 100)
for c, col in [(2, C["muted"]), (4, C["separate"]), (8, C["pooled"])]:
    axs[1].plot(rhos, [pooling_benefit(r * c * P.mu, P.mu, c)["percent_saved"] for r in rhos],
                color=col, lw=2, label=f"c = {c}")
axs[1].set_xlabel("utilisation ρ (share of time a lane is busy)")
axs[1].set_ylabel("wait saved by pooling (%)")
axs[1].set_title("Wait saved by pooling (%)")
axs[1].legend()
save(fig, "fig_pooling.png")

pooled_opt = optimize_staffing(prof, P, "pooled")
rows = []
for t in range(K):
    c = pooled_opt["lanes"][t]
    b = pooling_benefit(prof[t], P.mu, c)
    ls = lanes_needed("separate", prof[t], P.mu, P.c_max, P.metric, P.T, P.alpha)
    lp = lanes_needed("pooled", prof[t], P.mu, P.c_max, P.metric, P.T, P.alpha)
    rows.append({"block": t + 1, "start": labels[t], "lam": float(prof[t]), "c": c,
                 "Wq_sep": b["Wq_separate"], "Wq_pool": b["Wq_pooled"],
                 "min_saved": b["minutes_saved"], "pct_saved": b["percent_saved"],
                 "lanes_sep": ls, "lanes_pool": lp})
R["pooling_blocks"] = rows
sel = [0, 3, peak, 12, 21, 27, K - 1]
R["pooling_selected"] = [rows[i] for i in sel]
R["pooling_avg_pct"] = float(np.mean([r["pct_saved"] for r in rows]))
R["pooling_pax_weighted_sep"] = float(sum(r["lam"] * r["Wq_sep"] for r in rows) * P.delta)
R["pooling_pax_weighted_pool"] = float(sum(r["lam"] * r["Wq_pool"] for r in rows) * P.delta)
# Lanes saved with a binding target
Pt = P.copy(metric="tail", T=1.0, alpha=0.10)
ls_rows = []
for t in sel:
    s = lanes_needed("separate", prof[t], P.mu, 20, "tail", 1.0, 0.10)
    q = lanes_needed("pooled", prof[t], P.mu, 20, "tail", 1.0, 0.10)
    ls_rows.append({"start": labels[t], "lam": float(prof[t]), "sep": s, "pool": q, "saved": s - q})
R["lanes_saved_tail"] = ls_rows

# ------------------------------------------------------------------ 3 priority
lam_pk, c_pk = float(prof[peak]), pooled_opt["lanes"][peak]
r1 = fast_lane_metrics(lam_pk, 0.10, P.mu, P.mu, c_pk)
r2 = cobham_metrics(lam_pk, 0.10, P.mu, P.mu, c_pk)
R["priority"] = {"lam": lam_pk, "c": c_pk, "p": 0.10, "P1": r1, "P2": r2}
# P1 at a share that suits one fast lane
r1b = fast_lane_metrics(lam_pk, 1 / c_pk, P.mu, P.mu, c_pk)
R["priority"]["P1_p_equal"] = r1b
# fast priority passengers (mu_p = 3)
R["priority"]["P1_fastmu"] = fast_lane_metrics(lam_pk, 0.10, 3.0, P.mu, c_pk)
R["priority"]["P2_fastmu"] = cobham_metrics(lam_pk, 0.10, 3.0, P.mu, c_pk)
ps = np.linspace(0, 0.6, 121)
fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
y1 = [fast_lane_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_regular"] for x in ps]
y2 = [cobham_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_regular"] for x in ps]
yp = [fast_lane_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_ref"] for x in ps]
ys = [cobham_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_ref"] for x in ps]
f = lambda ys: [y if np.isfinite(y) and y < 50 else np.nan for y in ys]
axs[0].plot(ps, f(y1), color=C["fast_lane"], lw=2, label="P1 regular")
axs[0].plot(ps, f(y2), color=C["inlane_priority"], lw=2, label="P2 regular")
axs[0].plot(ps, f(yp), color=C["pooled"], lw=2, ls="--", label="No priority, pooled")
axs[0].plot(ps, f(ys), color=C["separate"], lw=2, ls="--", label="No priority, separate")
axs[0].set_ylim(0, 6)
axs[0].set_xlabel("priority share p (fraction of passengers)")
axs[0].set_ylabel("regular W_q (min)")
axs[0].set_title(f"Regular passengers, λ = {lam_pk:.2f}, c = {c_pk}")
axs[0].legend(fontsize=8)
z1 = [fast_lane_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_priority"] for x in ps]
z2 = [cobham_metrics(lam_pk, x, P.mu, P.mu, c_pk)["Wq_priority"] for x in ps]
axs[1].plot(ps, f(z1), color=C["fast_lane"], lw=2, label="P1 priority")
axs[1].plot(ps, f(z2), color=C["inlane_priority"], lw=2, label="P2 priority")
axs[1].plot(ps, f(yp), color=C["pooled"], lw=2, ls="--", label="No priority, pooled")
axs[1].set_ylim(0, 3)
axs[1].set_xlabel("priority share p (fraction of passengers)")
axs[1].set_ylabel("priority W_q (min)")
axs[1].set_title("Priority passengers")
axs[1].legend(fontsize=8)
save(fig, "fig_priority.png")

# ------------------------------------------------------------------ 4 staffing
cmp_df, rec, results = compare_designs(prof, P)
R["compare"] = cmp_df.to_dict("records")
R["recommended"] = rec
opt, base = results[rec]
R["plan"] = opt["plan"].assign(start=labels).to_dict("records")
R["plan_summary"] = {k: opt[k] for k in ("total_cost", "staff_cost", "wait_cost", "switch_cost",
                                         "pax_wait_min", "n_infeasible")}
R["plan_summary"]["lane_blocks"] = int(sum(opt["lanes"]))
R["baseline"] = {"lanes": base["lanes"][0], "total_cost": base["total_cost"],
                 "staff_cost": base["staff_cost"], "wait_cost": base["wait_cost"],
                 "pax_wait_min": base["pax_wait_min"]}
R["saving"] = saving(opt, base)
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(range(K), prof, color=C["ink"], lw=2, label="Arrival rate λ_t")
ax.step(range(K), np.array(opt["lanes"]) * P.mu, where="mid", color=C["pooled"], lw=2,
        label="Optimised capacity (lanes × μ)")
ax.step(range(K), np.array(base["lanes"]) * P.mu, where="mid", color=C["muted"], lw=1.5, ls="--",
        label=f"Fixed baseline ({base['lanes'][0]} lanes)")
for t in range(K):
    if t in (0, 4, peak, 11, 20, 24, 30, K - 1) or (t > 0 and opt["lanes"][t] != opt["lanes"][t - 1] and t % 2 == 0):
        pass
for t in (0, peak, 15, 25, K - 1):
    ax.annotate(f"{opt['lanes'][t]}", (t, opt["lanes"][t] * P.mu), textcoords="offset points",
                xytext=(0, 5), ha="center", fontsize=8, color=C["ink"])
ax.set_ylabel("passengers per minute")
ax.set_title("Optimised lanes per block (pooled queue) as screening capacity")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
xt(ax)
save(fig, "fig_staffing.png")

# switching cost: DP vs MIP
Psw = P.copy(Csw=300, max_step=2)
dp = optimize_staffing(prof, Psw, "pooled")
mip = optimize_staffing_mip(prof, Psw, "pooled")
R["switching"] = {"Csw": 300, "max_step": 2, "dp_cost": dp["total_cost"], "mip_cost": mip["total_cost"],
                  "lanes": dp["lanes"], "switch_cost": dp["switch_cost"], "no_switch_lanes": opt["lanes"]}
# binding tail target scenario
Pt = P.copy(metric="tail", T=2.0, alpha=0.05)
cmp_t, rec_t, res_t = compare_designs(prof, Pt)
R["tail_scenario"] = {"T": 2.0, "alpha": 0.05, "compare": cmp_t.to_dict("records"), "rec": rec_t,
                      "lanes": res_t[rec_t][0]["lanes"],
                      "saving": saving(*res_t[rec_t])}

# ------------------------------------------------------------------ 5 break-even
bt = breakeven_table(P.c_max, P.mu, P.Cs, P.Cw, P.delta, prof)
R["breakeven"] = bt.to_dict("records")
fig, ax = plt.subplots(figsize=(7.5, 3.2))
for c, col in [(3, C["separate"]), (5, C["pooled"]), (7, C["violet"])]:
    nb = net_benefit_curve(c, P.mu, P.Cs, P.Cw, P.delta)
    ax.plot(nb["lam"], nb["net_benefit"], color=col, lw=2, label=f"lane {c + 1} when {c} open")
    ls_ = bt.loc[bt.from_c == c, "lambda_star"].iloc[0]
    ax.axvline(ls_, color=col, lw=1, ls=":")
    ax.annotate(f"λ* = {ls_:.2f}", (ls_, 1200), rotation=90, fontsize=8, color=C["ink"],
                xytext=(-10, 0), textcoords="offset points")
ax.axhline(0, color=C["muted"], lw=1)
ax.plot(prof, np.zeros(K), "|", color=C["ink"], ms=12, mew=1.5, label="block arrival rates")
ax.set_ylim(-600, 3000)
ax.set_xlabel("arrival rate λ (passengers/min)")
ax.set_ylabel("net benefit (INR per 30-min block)")
ax.set_title("Net benefit of opening one more lane (pooled queue)")
ax.legend(fontsize=8, loc="upper left")
save(fig, "fig_breakeven.png")

# ------------------------------------------------------------------ 6 simulation
t0 = time.time()
sim_rows = []
blocks = [0, 3, peak, 21, 27]
for t in blocks:
    lam, c = float(prof[t]), opt["lanes"][t]
    e = evaluate_block("pooled", lam, c, P)
    s = simulate_block("pooled", lam, P.mu, c, T=1.0, reps=100, seed=100 + t)
    tail_a = mmc_metrics(lam, P.mu, c).tail(1.0)
    sim_rows.append({"block": t + 1, "start": labels[t], "design": "pooled", "lam": lam, "c": c,
                     "Wq": e["Wq"], "sim": s["mean_wait"], "lo": s["mean_wait_lo"], "hi": s["mean_wait_hi"],
                     "pass": passes(e["Wq"], s["mean_wait"], s["mean_wait_lo"], s["mean_wait_hi"]),
                     "tail_a": tail_a, "tail_s": s["tail"], "tail_lo": s["tail_lo"], "tail_hi": s["tail_hi"],
                     "tail_pass": passes(tail_a, s["tail"], s["tail_lo"], s["tail_hi"], abs_tol=0.005),
                     "util_a": lam / (c * P.mu), "util_s": s["util"]})
    if t == peak:
        hist = s["waits_sample"]
# other designs at the peak block
for d, c in [("separate", c_pk), ("fast_lane", c_pk), ("inlane_priority", c_pk)]:
    e = evaluate_block(d, lam_pk, c, P)
    s = simulate_block(d, lam_pk, P.mu, c, T=1.0, reps=100, seed=900)
    row = {"block": peak + 1, "start": labels[peak], "design": d, "lam": lam_pk, "c": c,
           "Wq": e["Wq"], "sim": s["mean_wait"], "lo": s["mean_wait_lo"], "hi": s["mean_wait_hi"],
           "pass": passes(e["Wq"], s["mean_wait"], s["mean_wait_lo"], s["mean_wait_hi"]),
           "tail_s": s["tail"], "util_a": e["rho"], "util_s": s["util"]}
    if d != "separate":
        row.update(Wq_p=e["Wq_priority"], sim_p=s["mean_wait_priority"],
                   Wq_r=e["Wq_regular"], sim_r=s["mean_wait_regular"],
                   pass_p=passes(e["Wq_priority"], s["mean_wait_priority"], s["mean_wait_priority_lo"], s["mean_wait_priority_hi"]),
                   pass_r=passes(e["Wq_regular"], s["mean_wait_regular"], s["mean_wait_regular_lo"], s["mean_wait_regular_hi"]))
    else:
        tail_a = mm1_split_metrics(lam_pk, P.mu, c).tail(1.0)
        row.update(tail_a=tail_a, tail_pass=passes(tail_a, s["tail"], s["tail_lo"], s["tail_hi"], abs_tol=0.005))
    sim_rows.append(row)
R["simulation"] = sim_rows

# stress tests at the peak block
stress_rows = []
base_s = simulate_block("pooled", lam_pk, P.mu, c_pk, T=1.0, reps=50, seed=500)
stress_rows.append({"test": "None (Poisson, all lanes open)", "design": "pooled",
                    "mean": base_s["mean_wait"], "lo": base_s["mean_wait_lo"], "hi": base_s["mean_wait_hi"],
                    "tail": base_s["tail"]})
for name, st, so in [("Group arrivals, mean group size 3", "groups", StressOptions(group_mean=3)),
                     ("Two-state rate, ±60%, 15-min states", "two_state", StressOptions()),
                     ("One lane closed 25% of each block", "lane_closure", StressOptions())]:
    s = simulate_block("pooled", lam_pk, P.mu, c_pk, T=1.0, reps=50, seed=500, stress=st, stress_opts=so)
    stress_rows.append({"test": name, "design": "pooled", "mean": s["mean_wait"], "lo": s["mean_wait_lo"],
                        "hi": s["mean_wait_hi"], "tail": s["tail"]})
rnd = simulate_block("separate", lam_pk, P.mu, c_pk, T=1.0, reps=50, seed=500)
jsq = simulate_block("separate", lam_pk, P.mu, c_pk, T=1.0, reps=50, seed=500, stress="jsq")
stress_rows.append({"test": "Separate lanes, random choice", "design": "separate", "mean": rnd["mean_wait"],
                    "lo": rnd["mean_wait_lo"], "hi": rnd["mean_wait_hi"], "tail": rnd["tail"]})
stress_rows.append({"test": "Separate lanes, join shortest queue", "design": "separate", "mean": jsq["mean_wait"],
                    "lo": jsq["mean_wait_lo"], "hi": jsq["mean_wait_hi"], "tail": jsq["tail"]})
R["stress"] = stress_rows
R["stress_ref"] = {"pooled": mmc_metrics(lam_pk, P.mu, c_pk).Wq, "separate": mm1_split_metrics(lam_pk, P.mu, c_pk).Wq}
R["sim_time_s"] = time.time() - t0

fig, axs = plt.subplots(1, 2, figsize=(9, 3.3))
names = [f"{r['start']}\n{r['design'].replace('inlane_priority', 'P2').replace('fast_lane', 'P1')}" for r in sim_rows]
x = np.arange(len(sim_rows))
axs[0].errorbar(x, [r["sim"] for r in sim_rows],
                yerr=[[r["sim"] - r["lo"] for r in sim_rows], [r["hi"] - r["sim"] for r in sim_rows]],
                fmt="o", color=C["pooled"], ms=5, capsize=3, label="simulated mean, 95% CI")
axs[0].plot(x, [r["Wq"] for r in sim_rows], "D", color=C["separate"], ms=5, label="analytical W_q",
            mfc="none", mew=1.5)
axs[0].set_xticks(x)
axs[0].set_xticklabels(names, fontsize=7)
axs[0].set_yscale("log")
axs[0].set_ylabel("mean wait W_q (min, log scale)")
axs[0].set_xlabel("block start time and queue design")
axs[0].set_title("Analytical vs simulated mean wait")
axs[0].legend(fontsize=8)
axs[1].hist(hist, bins=60, color=C["pooled"], edgecolor=IVORY, linewidth=0.5)
axs[1].set_yscale("log")
axs[1].set_xlabel("wait (min)")
axs[1].set_ylabel("number of passengers (log scale)")
axs[1].set_title(f"Simulated waits, peak block (c = {c_pk})")
save(fig, "fig_simulation.png")

# ------------------------------------------------------------------ 7 sensitivity
UNITS = {'arrival_scale': '× base profile', 'mu': 'passengers/min per lane', 'T': 'minutes', 'alpha': 'probability', 'cost_ratio': '× base C_w/C_s', 'p': 'fraction of passengers', 'cs2': 'squared coeff. of variation', 'c_max': 'lanes'}
ow = sens.all_one_way(prof, P, "pooled")
R["sens_oneway"] = ow.to_dict("records")
Pb = P.copy(metric="tail", T=2.0, alpha=0.05)     # a binding target, to show T and alpha
ow_b = pd_b = sens.all_one_way(prof, Pb, "pooled")
R["sens_oneway_binding"] = ow_b.to_dict("records")
ow_p1 = sens.one_way(prof, P, "fast_lane", "p")
R["sens_p_fastlane"] = ow_p1.to_dict("records")
fig, axs = plt.subplots(2, 4, figsize=(11, 5))
for ax, n in zip(axs.flat, sens.default_ranges(P)):
    d = ow[ow.parameter == n]
    db = ow_b[ow_b.parameter == n]
    ax.plot(d["value"], d["total_cost"] / 1000, color=C["pooled"], lw=2, marker="o", ms=3,
            label="mean target T=10")
    ax.plot(db["value"], db["total_cost"] / 1000, color=C["separate"], lw=2, marker="o", ms=3,
            label="tail target P(W>2)≤0.05")
    if n == "p":
        ax.plot(ow_p1["value"], ow_p1["total_cost"] / 1000, color=C["fast_lane"], lw=2, marker="o",
                ms=3, label="P1 fast lane")
        ax.legend(fontsize=6)
    if n in ("T", "alpha"):
        bvals = db["value"].values
    ax.set_title(sens.LABELS[n], fontsize=9)
    ax.set_xlabel(UNITS[n], fontsize=7)
    ax.set_ylabel("total cost ('000 INR/day)", fontsize=7)
    ax.tick_params(labelsize=7)

axs[0, 0].legend(fontsize=6)
fig.tight_layout()
save(fig, "fig_sens_oneway.png")

cc = cost_curve(float(prof[peak]), P, "pooled")
grid = sens.two_way_grid(prof, P, "pooled")
tor = sens.tornado(prof, P, "pooled")
tor_b = sens.tornado(prof, Pb, "pooled")
R["cost_curve"] = cc.replace(np.inf, None).to_dict("records")
R["grid"] = grid.to_dict("records")
R["tornado"] = tor.to_dict("records")
R["tornado_binding"] = tor_b.to_dict("records")
fig, axs = plt.subplots(1, 3, figsize=(12, 3.5), gridspec_kw={"width_ratios": [1, 1, 1.2]})
ax = axs[0]
ax.plot(cc["c"], cc["staff_cost"], color=C["pooled"], lw=2, label="staffing")
ax.plot(cc["c"], cc["wait_cost"].replace(np.inf, np.nan), color=C["separate"], lw=2, label="waiting")
ax.plot(cc["c"], cc["total_cost"].replace(np.inf, np.nan), color=C["ink"], lw=2.5, marker="o", ms=4, label="total")
ax.set_ylim(0, 9000)
ax.set_xlabel("open lanes c")
ax.set_ylabel("cost (INR per 30-min block)")
ax.set_title(f"Peak block cost trade-off (λ = {prof[peak]:.2f})")
ax.legend(fontsize=8)
piv = grid.pivot(index="arrival_scale", columns="Cw", values="total_cost")
im = axs[1].imshow(piv.values / 1000, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("ivblue", [IVORY, BLUE_LT, BLUE]), aspect="auto", origin="lower")
axs[1].set_xticks(range(len(piv.columns)))
axs[1].set_xticklabels([f"{v:g}" for v in piv.columns])
axs[1].set_yticks(range(len(piv.index)))
axs[1].set_yticklabels([f"{v:g}×" for v in piv.index])
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j] / 1000
        axs[1].text(j, i, f"{v:.0f}k", ha="center", va="center", fontsize=8,
                    color=IVORY if v > np.nanmean(piv.values) / 1000 else C["ink"])
axs[1].set_xlabel("C_w (INR per passenger-minute)")
axs[1].set_ylabel("arrival scale (× base profile)")
axs[1].set_title("Total cost ('000 INR)")
axs[1].grid(False)
t2 = tor[tor["swing"] > 0].sort_values("swing")
y = np.arange(len(t2))
b0 = t2["base_cost"].iloc[0]
axs[2].barh(y, (t2["cost_low"] - b0) / 1000, color=C["pooled"], height=0.6, label="low value")
axs[2].barh(y, (t2["cost_high"] - b0) / 1000, color=C["separate"], height=0.6, label="high value")
axs[2].set_yticks(y)
axs[2].set_yticklabels(t2["label"], fontsize=8)
axs[2].axvline(0, color=C["ink"], lw=1)
axs[2].set_xlabel("change in total cost ('000 INR per day)")
axs[2].set_title("Tornado (inputs with an effect)")
axs[2].legend(fontsize=8)
fig.tight_layout()
save(fig, "fig_sens_tradeoff.png")


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return None if not np.isfinite(o) else o
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(clean(R), fh, indent=1)
print("done", R["sim_time_s"])
