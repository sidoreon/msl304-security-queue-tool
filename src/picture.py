"""Animated picture of the checkpoint: lanes at the top, passengers queueing below them.

Colours: regular passengers clay orange, priority passengers light blue,
a lane that is screening someone is filled US Open blue.
"""
import plotly.graph_objects as go

IVORY, INK, CREAM = "#FBF8EF", "#1E2633", "#F1EADA"
BLUE, BLUE_LT, CLAY, GREY = "#1D4F91", "#6E9BD1", "#C65A2E", "#8C8577"
GAP = 0.44       # space between people in a line
SHOW = 11        # people drawn in one column before "+n more"


def _people(snap, groups):
    rx, ry, px, py, lx, ly, lt = [], [], [], [], [], [], []
    for (lanes, _, _), line in zip(groups, snap["lines"]):
        mid = sum(lanes) / len(lanes)
        cols = 3 if len(lanes) >= 3 else 1          # a long shared line snakes in 3 rows
        drawn = min(len(line), SHOW * cols)
        for k in range(drawn):
            row, col = divmod(k, cols)
            x = mid + (col - (cols - 1) / 2) * 0.42
            y = -0.5 - row * GAP
            (px if line[k] else rx).append(x)
            (py if line[k] else ry).append(y)
        if len(line) > drawn:
            lx.append(mid); ly.append(-0.5 - SHOW * GAP - 0.15); lt.append(f"+{len(line) - drawn} more")
    return (rx, ry), (px, py), (lx, ly, lt)


def _frame_traces(snap, groups, c):
    (rx, ry), (px, py), (lx, ly, lt) = _people(snap, groups)
    busy = snap["busy"]
    reg = [k for k, pri in busy.items() if not pri]
    pri = [k for k, pri in busy.items() if pri]
    waiting = sum(len(l) for l in snap["lines"])
    return [
        go.Scatter(x=reg + pri, y=[0.3] * (len(reg) + len(pri)), mode="markers",
                   name="being screened", hoverinfo="skip",
                   marker=dict(size=[15] * len(reg) + [15] * len(pri),
                               symbol=["circle"] * len(reg) + ["star"] * len(pri),
                               color=BLUE, line=dict(color=IVORY, width=1.5))),
        go.Scatter(x=rx, y=ry, mode="markers", name="waiting", hoverinfo="skip",
                   marker=dict(size=13, color=CLAY, line=dict(color=IVORY, width=1.5))),
        go.Scatter(x=px, y=py, mode="markers", name="priority passenger", hoverinfo="skip",
                   marker=dict(size=13, symbol="star", color=BLUE_LT, line=dict(color=BLUE, width=1.5))),
        go.Scatter(x=lx, y=ly, mode="text", text=lt, showlegend=False, hoverinfo="skip",
                   textfont=dict(color=INK, size=12)),
        go.Scatter(x=[-0.65], y=[1.55], mode="text", showlegend=False, hoverinfo="skip",
                   textposition="middle right", textfont=dict(color=INK, size=13),
                   text=[f"<b>{waiting} in line</b> · average wait so far {snap['avg_wait']:.1f} min"]),
    ]


def queue_animation(snaps, groups, c, title, height=430, show_priority=False):
    fig = go.Figure(data=_frame_traces(snaps[0], groups, c),
                    frames=[go.Frame(data=_frame_traces(s, groups, c), name=str(i))
                            for i, s in enumerate(snaps)])
    fast = {lanes[0] for lanes, _, who in groups if who == "priority"}
    for k in range(c):
        fig.add_shape(type="rect", x0=k - 0.3, x1=k + 0.3, y0=0.08, y1=0.52, layer="below",
                      line=dict(color=BLUE, width=5 if k in fast else 2), fillcolor="#E8EEF7")
        if k in fast:
            fig.add_annotation(x=k, y=1.0, yanchor="bottom", text="<b>fast lane</b>", showarrow=False,
                               font=dict(size=11, color=BLUE))
    # one officer (small black dot) behind every lane desk
    fig.add_trace(go.Scatter(x=list(range(c)), y=[0.68] * c, mode="markers", name="staff",
                             hoverinfo="skip", marker=dict(size=8, color="#000000")))
    if not show_priority:
        fig.data[2].showlegend = False
    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=15, color=INK)),
        height=height, margin=dict(l=5, r=5, t=60, b=5),
        paper_bgcolor=IVORY, plot_bgcolor=IVORY, font=dict(color=INK),
        xaxis=dict(visible=False, range=[-0.75, max(c - 1, 3) + 0.75], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.5 - SHOW * GAP - 0.45, 1.75], fixedrange=True),
        legend=dict(orientation="h", x=0, y=-0.02, yanchor="top", font=dict(size=11)),
        updatemenus=[dict(type="buttons", direction="left", x=1, xanchor="right", y=-0.01, yanchor="top",
                          bgcolor=CREAM, bordercolor=GREY, font=dict(color=INK, size=12),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=110, redraw=True),
                                                         transition=dict(duration=0), fromcurrent=True)]),
                                   dict(label="❚❚", method="animate",
                                        args=[[None], dict(mode="immediate", frame=dict(duration=0))])])],
    )
    return fig
