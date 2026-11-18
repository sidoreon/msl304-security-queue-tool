"""Security checkpoint planner (Streamlit).

Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.queueing import mmc_metrics, mm1_split_metrics
from src.priority import fast_lane_metrics, cobham_metrics
from src.staffing import Params, DESIGNS, optimize_staffing, baseline_plan, saving, compare_designs
from src.breakeven import breakeven_table
from src.sample_data import sample_profile
from src.validation import validate_inputs
from src import animation as anim
from src.picture import queue_animation

st.set_page_config(page_title="Queueing-Based Design of Airport Security Checkpoints", layout="wide")

IVORY, INK, GRID = "#FBF8EF", "#1E2633", "#E6E0D0"
BLUE, BLUE_LT, CLAY, CLAY_LT, GREY = "#1D4F91", "#6E9BD1", "#C65A2E", "#E3A47E", "#8C8577"
NAMES = {"pooled": "One shared line", "separate": "A line per lane",
         "fast_lane": "Fast lane for priority", "inlane_priority": "Priority first in every line"}
COLOURS = {"pooled": BLUE, "separate": CLAY, "fast_lane": BLUE_LT, "inlane_priority": CLAY_LT}


def style(fig, title="", height=360, ytitle=None):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor=IVORY, plot_bgcolor=IVORY,
                      font=dict(color=INK, size=13), title=dict(text=title, x=0.01),
                      margin=dict(l=10, r=10, t=50 if title else 20, b=10),
                      legend=dict(orientation="h", y=1.0, x=0.99, xanchor="right", yanchor="bottom"),
                      hovermode="x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, title=ytitle, zeroline=False)
    return fig


def minutes(x):
    return "too long (queue keeps growing)" if not np.isfinite(x) else f"{x:.1f} min"


@st.cache_data(show_spinner=False)
def animate(design, lam, mu, c, p, mu_p, seed=1):
    """Simulate 40 minutes and show minutes 10-40, so the line has had time to build up."""
    pax = anim.make_passengers(lam, mu, 40, p=p, mu_p=mu_p, seed=seed)
    times = np.arange(10, 40, 0.3)
    snaps, groups = anim.snapshots(design, pax, c, times)
    return snaps, groups


def show_animation(design, lam, mu, c, p, mu_p, title, height=430):
    snaps, groups = animate(design, lam, mu, c, p, mu_p)
    fig = queue_animation(snaps, groups, c, title, height=height,
                          show_priority=design in ("fast_lane", "inlane_priority"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ------------------------------------------------------------------ inputs
if "profile" not in st.session_state:
    st.session_state.profile = sample_profile()

sb = st.sidebar
sb.header("Checkpoint inputs")
mu = sb.number_input("Passengers one lane screens per minute", 0.2, 20.0, 2.0, 0.1)
c_max = int(sb.number_input("Lanes available", 1, 40, 10, 1))
Cs = sb.number_input("Cost of one lane for 30 min (INR)", 0.0, 1e6, 400.0, 50.0)
Cw = sb.number_input("Cost of one minute of a passenger's time (INR)", 0.0, 1e4, 5.0, 0.5)
T = sb.number_input("Longest acceptable average wait (min)", 0.1, 120.0, 10.0, 0.5)
sb.header("Priority passengers")
p = sb.slider("Share of priority passengers", 0.0, 0.5, 0.10, 0.01)
mu_p = sb.number_input("Priority passengers screened per minute", 0.2, 20.0, float(mu), 0.1)
with sb.expander("More settings"):
    cs2 = st.number_input("Screening time variability (1 = random, 0 = always the same)", 0.0, 5.0, 1.0, 0.1)
    Csw = st.number_input("Cost of opening an extra lane (INR)", 0.0, 1e5, 0.0, 50.0)

P = Params(delta=30, mu=mu, cs2=cs2, c_max=c_max, Cs=Cs, Cw=Cw, T=T, p=p, mu_p=mu_p, Csw=Csw)
prof = st.session_state.profile
times = list(prof["start"])
lams = tuple(float(x) for x in pd.to_numeric(prof["lam"], errors="coerce").fillna(-1))
errors, warnings = validate_inputs(lams, P)

st.title("Queueing-Based Design of Airport Security Checkpoints")
st.markdown("**MSL304 Course Project**  \nSohana Kumar (2024CS10251)  \nSidharth Valsan (2024MT10433)")

tabs = st.tabs(["1 · Problem", "2 · Pooled vs Separate Queues", "3 · Priority Lane Designs",
                "4 · Staffing Plan", "5 · Simulation Check", "6 · Conclusions"])

# ================================================================== 1 the day
with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(go.Scatter(x=times, y=lams, mode="lines", line=dict(color=BLUE, width=3),
                                   fill="tozeroy", fillcolor="rgba(29,79,145,0.12)", name="arrivals"))
        style(fig, "Passengers arriving per minute", 340)
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### Problem")
        st.markdown(
            "Passengers arrive faster in the morning and evening rush. Each open lane costs money, "
            "and every minute a passenger waits also costs money.\n\n"
            "For every half hour, the planner picks the number of lanes that gives the **lowest total "
            "cost**, as long as the average wait stays under your limit. It also compares four ways "
            "of arranging the queue.")
        st.info("The arrival numbers and costs are made-up examples. Change them on the left or below.")
    with st.expander("Edit the arrival numbers"):
        edited = st.data_editor(prof, hide_index=True, width="stretch", height=300,
                                column_config={"lam": st.column_config.NumberColumn("passengers per minute",
                                                                                    min_value=0.0, step=0.1),
                                               "block": None},
                                key="editor")
        if not edited.equals(prof):
            st.session_state.profile = edited.reset_index(drop=True)
            st.rerun()
        if st.button("Reset to the example day"):
            st.session_state.profile = sample_profile()
            st.rerun()
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)

if errors:
    st.stop()

compare_df, recommended, results = compare_designs(np.array(lams), P)
peak = int(np.argmax(lams))

# ================================================================== 2 one line or many
with tabs[1]:
    st.markdown("Same lanes, same passengers. The only difference is how the queue is arranged.")
    a, b = st.columns(2)
    t_pick = a.select_slider("Time of day", options=times, value=times[peak])
    lam = lams[times.index(t_pick)]
    c = b.slider("Open lanes", 1, c_max, int(results["pooled"][0]["lanes"][times.index(t_pick)]))
    w_shared, w_sep = mmc_metrics(lam, mu, c).Wq, mm1_split_metrics(lam, mu, c).Wq
    k1, k2, k3 = st.columns(3)
    k1.metric("Average wait, one shared line", minutes(w_shared))
    k2.metric("Average wait, a line per lane", minutes(w_sep))
    if np.isfinite(w_sep) and w_sep > 0:
        k3.metric("Time saved by sharing", f"{100 * (w_sep - w_shared) / w_sep:.0f}%")
    if lam >= c * mu:
        st.error("These lanes cannot keep up: more people arrive than can be screened. Open more lanes.")
    else:
        g1, g2 = st.columns(2)
        with g1:
            show_animation("pooled", lam, mu, c, 0.0, mu, "One shared line")
        with g2:
            show_animation("separate", lam, mu, c, 0.0, mu, "A line per lane (passengers pick at random)")
        st.caption("Press ▶ Play. With a line per lane, one lane can sit empty while people wait at "
                   "the next one. A shared line never lets that happen.")

# ================================================================== 3 priority
with tabs[2]:
    st.markdown("Two ways to let priority passengers (shown as stars) through faster. Each small black dot is an officer at a lane.")
    a, b = st.columns(2)
    t_pick3 = a.select_slider("Time of day ", options=times, value=times[peak])
    lam3 = lams[times.index(t_pick3)]
    c3 = b.slider("Open lanes ", 2, c_max, max(2, int(results["pooled"][0]["lanes"][times.index(t_pick3)])))
    if lam3 >= c3 * mu:
        st.error("These lanes cannot keep up. Open more lanes.")
    else:
        r1 = fast_lane_metrics(lam3, p, mu_p, mu, c3, cs2)
        r2 = cobham_metrics(lam3, p, mu_p, mu, c3, cs2)
        g1, g2 = st.columns(2)
        with g1:
            show_animation("fast_lane", lam3, mu, c3, p, mu_p, "A separate fast lane")
            st.metric("Priority passengers wait", minutes(r1["Wq_priority"]))
            st.metric("Everyone else waits", minutes(r1["Wq_regular"]),
                      delta=f"{r1['regular_change']:+.2f} min vs one shared line" if np.isfinite(r1["regular_change"]) else None,
                      delta_color="inverse")
        with g2:
            show_animation("inlane_priority", lam3, mu, c3, p, mu_p, "Priority goes first in every line")
            st.metric("Priority passengers wait", minutes(r2["Wq_priority"]))
            st.metric("Everyone else waits", minutes(r2["Wq_regular"]),
                      delta=f"{r2['regular_change']:+.2f} min vs a line per lane" if np.isfinite(r2["regular_change"]) else None,
                      delta_color="inverse")
        st.caption("Priority helps priority passengers, but the time comes from everyone else. "
                   "A whole fast lane for a small group often sits half empty.")

# ================================================================== 4 plan
with tabs[3]:
    design = st.radio("Queue layout", DESIGNS, index=DESIGNS.index(recommended), horizontal=True,
                      format_func=lambda d: NAMES[d] + (" ★" if d == recommended else ""))
    opt, base = results[design]
    sv = saving(opt, base)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cost per day, planned lanes", f"₹{opt['total_cost']:,.0f}")
    k2.metric(f"Cost per day, {base['lanes'][0]} lanes all day", f"₹{base['total_cost']:,.0f}")
    k3.metric("Saving", f"{100 * sv:.0f}%")
    k4.metric("Longest average wait", minutes(opt["plan"]["Wq"].max()))
    if opt["n_infeasible"]:
        st.warning(f"In {opt['n_infeasible']} half-hour(s) even {c_max} lanes are not enough to meet the wait limit.")

    lanes = np.array(opt["lanes"])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=times, y=lanes, name="lanes open", marker=dict(color=BLUE, line=dict(width=0)),
                         opacity=0.85))
    fig.add_trace(go.Scatter(x=times, y=np.array(lams) / mu, name="lanes needed just to keep up",
                             line=dict(color=CLAY, width=3)))
    fig.add_trace(go.Scatter(x=times, y=base["lanes"], name="fixed plan", mode="lines",
                             line=dict(color=GREY, width=2, dash="dash")))
    style(fig, "Lanes open in each half hour", 380, "lanes")
    fig.update_layout(bargap=0.15)
    st.plotly_chart(fig, width="stretch")
    st.caption("Orange line: arrivals divided by lane speed, the bare minimum. The plan always stays "
               "above it, with a little spare so the line stays short.")

    t_pick4 = st.select_slider("See the queue at", options=times, value=times[peak])
    i4 = times.index(t_pick4)
    if np.isfinite(opt["plan"]["Wq"][i4]):
        show_animation(design, lams[i4], mu, int(lanes[i4]), p if design in ("fast_lane", "inlane_priority") else 0.0,
                       mu_p, f"{t_pick4}: {lanes[i4]} lanes open, {lams[i4]:.1f} passengers per minute", height=400)

    with st.expander("Compare all four layouts"):
        tbl = compare_df[["design", "total_cost", "baseline_cost", "saving_pct", "peak_lanes"]].copy()
        tbl["design"] = tbl["design"].map(NAMES)
        tbl.columns = ["Layout", "Planned cost (₹)", "Fixed-lane cost (₹)", "Saving (%)", "Most lanes"]
        st.dataframe(tbl.round(0), hide_index=True, width="stretch")
    with st.expander("Plan as a table"):
        tbl = pd.DataFrame({"Time": times, "Passengers/min": np.round(lams, 1), "Lanes": lanes,
                            "Average wait (min)": opt["plan"]["Wq"].round(2),
                            "Cost (₹)": opt["plan"]["cost"].round(0)})
        st.dataframe(tbl, hide_index=True, width="stretch", height=300)
        st.download_button("Download plan (CSV)", tbl.to_csv(index=False).encode(), "plan.csv", "text/csv")

# ================================================================== 5 checks
with tabs[4]:
    st.markdown("#### Formula against simulation")
    st.markdown("We simulate thousands of passengers and compare the average wait with the formula.")
    if st.button("Run the check", type="primary"):
        rows = []
        for t in sorted({peak, int(np.argmin(lams)), times.index("18:00") if "18:00" in times else peak}):
            lam_t, c_t = lams[t], int(results["pooled"][0]["lanes"][t])
            pax = anim.make_passengers(lam_t, mu, 30000, p=p, mu_p=mu_p, seed=7)
            for d in DESIGNS:
                if d == "fast_lane" and c_t < 2:
                    continue
                f = {"pooled": lambda: mmc_metrics(lam_t, mu, c_t, cs2).Wq,
                     "separate": lambda: mm1_split_metrics(lam_t, mu, c_t, cs2=cs2).Wq,
                     "fast_lane": lambda: fast_lane_metrics(lam_t, p, mu_p, mu, c_t, cs2)["Wq"],
                     "inlane_priority": lambda: cobham_metrics(lam_t, p, mu_p, mu, c_t, cs2)["Wq"]}[d]()
                sim = anim.average_waits(d, pax, c_t, warmup=500)["all"]
                rows.append({"Time": times[t], "Lanes": c_t, "Layout": NAMES[d],
                             "Formula (min)": round(f, 3), "Simulation (min)": round(sim, 3)})
        st.session_state.check = pd.DataFrame(rows)
    if "check" in st.session_state:
        df = st.session_state.check
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["Time"] + " · " + df["Layout"], y=df["Formula (min)"], name="formula",
                             marker_color=BLUE))
        fig.add_trace(go.Bar(x=df["Time"] + " · " + df["Layout"], y=df["Simulation (min)"], name="simulation",
                             marker_color=CLAY))
        style(fig, "Average wait: formula and simulation side by side", 360, "minutes")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch")

    st.markdown("#### Break-even arrival rates")
    bt = breakeven_table(min(c_max, 10), mu, Cs, Cw, 30)
    bt = bt[bt["lambda_star"].notna()]
    fig = go.Figure(go.Bar(x=[f"{a}→{b}" for a, b in zip(bt["from_c"], bt["to_c"])],
                           y=bt["lambda_star"], marker_color=BLUE,
                           text=[f"{v:.1f}" for v in bt["lambda_star"]], textposition="outside"))
    style(fig, "Open the next lane once more than this many passengers arrive per minute", 320,
          "passengers per minute")
    st.plotly_chart(fig, width="stretch")
    st.caption("Rule of thumb with these costs: open another lane when the open lanes are about "
               "80–87% busy.")

# ================================================================== 6 conclusions
with tabs[5]:
    st.markdown("#### Conclusions")
    st.caption("Worked out from the inputs on the left. Change them and these update.")

    lam_pk = lams[peak]
    c_pk = int(results["pooled"][0]["lanes"][peak])
    w_pool, w_sep = mmc_metrics(lam_pk, mu, c_pk).Wq, mm1_split_metrics(lam_pk, mu, c_pk).Wq
    opt_p, base_p = results["pooled"]
    r1 = fast_lane_metrics(lam_pk, p, mu_p, mu, c_pk, cs2) if c_pk >= 2 else None
    r2 = cobham_metrics(lam_pk, p, mu_p, mu, c_pk, cs2)
    bt = breakeven_table(min(c_max, 10), mu, Cs, Cw, 30)
    util = (bt["lambda_star"] / (bt["from_c"] * mu)).dropna()

    def card(col, number, text):
        col.markdown(
            f"<div style='background:#F1EADA;border-radius:12px;padding:18px 20px;height:100%'>"
            f"<div style='font-size:{34 if len(number) < 10 else 22}px;font-weight:700;color:{BLUE};line-height:1.2'>{number}</div>"
            f"<div style='color:{INK};font-size:15px;margin-top:4px'>{text}</div></div>",
            unsafe_allow_html=True)

    a, b, c_, d = st.columns(4)
    card(a, f"{100 * (w_sep - w_pool) / w_sep:.0f}%" if np.isfinite(w_sep) and w_sep > 0 else "–",
         f"less waiting with one shared line at the peak ({times[peak]}, {c_pk} lanes)")
    card(b, f"{100 * saving(opt_p, base_p):.0f}%",
         f"cheaper per day than keeping {base_p['lanes'][0]} lanes open all day")
    card(c_, f"{util.min() * 100:.0f}–{util.max() * 100:.0f}%" if len(util) else "–",
         "busy: open the next lane once the open lanes reach this level")
    card(d, NAMES[recommended], "is the cheapest way to arrange the queue")
    st.write("")

    # 1 pooling
    st.markdown("##### 1. One shared line beats a line per lane")
    l, r = st.columns([2, 3])
    with l:
        fig = go.Figure(go.Bar(x=["One shared line", "A line per lane"], y=[w_pool, w_sep],
                               marker_color=[BLUE, CLAY], text=[minutes(w_pool), minutes(w_sep)],
                               textposition="outside"))
        style(fig, f"Average wait at {times[peak]} with {c_pk} lanes", 300, "minutes")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with r:
        st.markdown(
            f"- With the same lanes and the same passengers, a shared line cuts the average wait from "
            f"**{minutes(w_sep)}** to **{minutes(w_pool)}**.\n"
            "- With separate lines, one lane can stand empty while people queue at the next. "
            "A shared line never lets a lane sit idle while someone is waiting.\n"
            "- If the space only allows separate lines, pointing passengers to the shortest one "
            "recovers most of the difference.")

    # 2 cost
    st.markdown("##### 2. Matching lanes to the rush saves money")
    l, r = st.columns([2, 3])
    with l:
        order = list(compare_df.sort_values("total_cost")["design"])
        fig = go.Figure()
        fig.add_trace(go.Bar(y=[NAMES[x] for x in order], x=[results[x][1]["total_cost"] for x in order],
                             orientation="h", name="same lanes all day", marker_color=GREY))
        fig.add_trace(go.Bar(y=[NAMES[x] for x in order], x=[results[x][0]["total_cost"] for x in order],
                             orientation="h", name="lanes matched to demand",
                             marker_color=[COLOURS[x] for x in order]))
        style(fig, "Cost per day (₹)", 320)
        fig.update_layout(barmode="group", hovermode="y unified")
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, width="stretch")
    with r:
        st.markdown(
            f"- Opening {min(opt_p['lanes'])} lanes in quiet times and up to {max(opt_p['lanes'])} in the rush costs "
            f"**₹{opt_p['total_cost']:,.0f}** a day, against **₹{base_p['total_cost']:,.0f}** for "
            f"{base_p['lanes'][0]} lanes all day.\n"
            f"- The average wait in every half hour stays at or below **{minutes(opt_p['plan']['Wq'].max())}**.\n"
            "- At the peak the plan adds one more lane than the bare minimum, because the waiting time it "
            "saves is worth more than the lane costs.")

    # 3 priority
    st.markdown("##### 3. Priority lanes move waiting, they do not remove it")
    l, r = st.columns([2, 3])
    with l:
        cats = ["Priority passengers", "Everyone else"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cats, y=[w_pool, w_pool], name="One shared line", marker_color=BLUE))
        if r1 is not None:
            fig.add_trace(go.Bar(x=cats, y=[r1["Wq_priority"], r1["Wq_regular"]], name="Fast lane",
                                 marker_color=BLUE_LT))
        fig.add_trace(go.Bar(x=cats, y=[r2["Wq_priority"], r2["Wq_regular"]], name="Priority first in every line",
                             marker_color=CLAY_LT))
        style(fig, f"Average wait at {times[peak]} (minutes)", 340)
        fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.15, x=0, xanchor="left", yanchor="top"))
        st.plotly_chart(fig, width="stretch")
    with r:
        msg = [f"- Letting priority passengers go first in every line cuts their wait to **{minutes(r2['Wq_priority'])}**, "
               f"but everyone else waits **{r2['regular_change']:+.2f} min** longer."]
        if r1 is not None:
            msg.append(f"- A separate fast lane for {p:.0%} of passengers is often half empty; everyone else waits "
                       f"**{r1['regular_change']:+.2f} min** longer than in one shared line.")
        msg.append("- A fast lane makes sense only if it is sized to the number of priority passengers, "
                   "or if they are quicker to screen.")
        st.markdown("\n".join(msg))

    # 4 rule + checks
    st.markdown("##### 4. A simple rule, and the maths holds up")
    l, r = st.columns([2, 3])
    with l:
        if len(util):
            fig = go.Figure(go.Scatter(x=bt.loc[util.index, "from_c"], y=util * 100, mode="lines+markers",
                                       line=dict(color=BLUE, width=3), marker=dict(size=9, color=CLAY)))
            style(fig, "How busy lanes get before the next one pays off", 300, "% busy")
            fig.update_xaxes(title="lanes already open")
            st.plotly_chart(fig, width="stretch")
    with r:
        st.markdown(
            f"- Supervisors can follow one rule: when the open lanes are about "
            f"**{util.min() * 100:.0f}–{util.max() * 100:.0f}% busy**, open another.\n"
            "- A simulation of thousands of passengers gives the same average waits as the formulas "
            "(see Simulation Check).\n"
            "- The formulas assume passengers arrive randomly but steadily. When they come in bursts, "
            "such as after a flight lands, waits get much longer, so keep a spare lane ready.")

    st.markdown("##### Recommendations")
    recs = ["Use one shared line feeding all lanes.",
            "Open and close lanes through the day to follow the arrival curve.",
            f"Open another lane when the open lanes are about {util.min() * 100:.0f}–{util.max() * 100:.0f}% busy." if len(util) else "Open another lane before the open ones are nearly full.",
            "Size any fast lane to the share of priority passengers.",
            "Keep a spare lane for sudden rushes."]
    cols = st.columns(len(recs))
    for k, (col, txt) in enumerate(zip(cols, recs), start=1):
        col.markdown(
            f"<div style='border:2px solid {BLUE};border-radius:12px;padding:14px;height:100%;background:{IVORY}'>"
            f"<div style='font-size:26px;font-weight:700;color:{CLAY}'>{k}</div>"
            f"<div style='color:{INK};font-size:14px'>{txt}</div></div>", unsafe_allow_html=True)
    st.write("")
    st.caption("All arrival numbers and costs are made-up examples, not measurements from a real airport.")
