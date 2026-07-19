import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Litmus - how accurate is your accuracy?", layout="wide")

path = Path(__file__).parent.parent / "results.json"
if not path.exists():
    path = Path("results.json")
R = json.loads(path.read_text())

base, sc, proxy = R["baseline"], R["shortcut"], R["proxy"]
mp_a, mp_m, leak = R["multiplicity_adult"], R["multiplicity_market"], R["leakage"]
compas = R["compas"]

RED = "#ff4d4d"
GRAY = "#39414f"
MUTED = "#8b94a3"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

.stApp { background: radial-gradient(1200px 600px at 20% -10%, #1a1420 0%, #0e1117 55%) fixed; }
html, body, [class*="css"], .stMarkdown, p, li { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1150px; padding-top: 4.5rem; padding-bottom: 5rem; }

.hero-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; font-weight: 500;
    color: #ff4d4d; letter-spacing: 0.22em; text-transform: uppercase; margin-bottom: 1rem; }
.hero-title { font-family: 'Instrument Serif', serif; font-size: 5.6rem; line-height: 0.95;
    color: #ffffff; margin-bottom: 1.4rem; letter-spacing: -0.015em; }
.hero-title em { color: #ff4d4d; font-style: italic; }
.hero-sub { font-size: 1.25rem; color: #c3c9d2; line-height: 1.7; max-width: 640px; margin-bottom: 0.7rem; }
.hero-meta { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #5d6673;
    letter-spacing: 0.03em; margin-bottom: 1.6rem; }

.metric-card { background: linear-gradient(180deg, #171c26 0%, #11151d 100%);
    border: 1px solid #242c3a; border-radius: 14px; padding: 1.35rem 1.4rem; transition: border-color .2s; }
.metric-card:hover { border-color: #3b4657; }
.metric-label { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.1em; color: #7e8797; margin-bottom: 0.5rem; }
.metric-value { font-family: 'Instrument Serif', serif; font-size: 2.6rem; color: #fff; line-height: 1; }
.metric-delta-up { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #4ade80; margin-top: 0.5rem; }
.metric-delta-down { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #ff4d4d; margin-top: 0.5rem; }
.metric-delta-flat { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #7e8797; margin-top: 0.5rem; }

.section-num { font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 500;
    color: #ff4d4d; letter-spacing: 0.18em; margin-bottom: 0.3rem; }
h3 { font-family: 'Instrument Serif', serif !important; font-size: 2.3rem !important;
    font-weight: 400 !important; letter-spacing: 0 !important;
    margin: 0 0 0.6rem 0 !important; padding: 0 !important; color: #fff !important; }
.section-body { font-size: 1.02rem; color: #b9c0cb; line-height: 1.75; max-width: 680px; margin-bottom: 1.4rem; }
.section-body code { background: #1c2330; padding: 0.1em 0.4em; border-radius: 5px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.85em; color: #e8b4b8; }

[data-testid="stPlotlyChart"] { background: linear-gradient(180deg, #141924 0%, #10141c 100%);
    border: 1px solid #212938; border-radius: 16px; padding: 1.4rem 1.2rem 0.8rem 1.2rem; }
.spacer { height: 4.5rem; }
.footer { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #7e8797;
    border-top: 1px solid #1f2530; padding-top: 2rem; margin-top: 4rem; }
.footer a { color: #d6dbe3; text-decoration: none; border-bottom: 1px solid #3b4657; }

[data-testid="stPageLink"] a {
    background: linear-gradient(180deg, #1d2430 0%, #161c26 100%);
    border: 1px solid #2f3947; border-radius: 10px;
    padding: 0.7rem 1.2rem !important; display: inline-flex;
    font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important;
    color: #e8eaed !important; transition: all .2s;
}
[data-testid="stPageLink"] a:hover { border-color: #ff4d4d; background: #212a38; }
</style>
""", unsafe_allow_html=True)

PLOT = dict(
    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13.5, color="#c3c9d2"),
    margin=dict(t=18, b=12, l=10, r=10),
    hoverlabel=dict(bgcolor="#1c2330", bordercolor="#3b4657",
                    font=dict(family="JetBrains Mono, monospace", size=13, color="#e8eaed")),
)
MONO = dict(family="JetBrains Mono, monospace")


def substantial_pct(frac, band):
    f = np.array(frac)
    return float(((f >= band) & (f <= 1 - band)).mean() * 100)


def card(col, label, value, delta=None, kind="flat"):
    d = f'<div class="metric-delta-{kind}">{delta}</div>' if delta else ""
    col.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div>'
                 f'<div class="metric-value">{value}</div>{d}</div>', unsafe_allow_html=True)


# ── hero ────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-eyebrow">A model-auditing toolkit</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">How accurate is<br>your <em>accuracy?</em></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Litmus runs five audits against models that look fine on paper. '
    'Every detector is validated on planted ground truth, then pointed at real data, '
    'from census income to stock prediction to criminal risk scores.</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-meta">XGBOOST · ADULT/CENSUS 45K · SPY DAILY 2010–2024 · COMPAS 5K DEFENDANTS · REPRODUCIBLE</div>',
    unsafe_allow_html=True)

st.page_link("pages/1_Audit_your_model.py", label="Audit your own model →")
st.markdown('<div style="height:2.4rem"></div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4, gap="medium")
card(c1, "honest baseline", f"{base['acc']:.1%}")
card(c2, "with planted leak", f"{sc['acc']:.1%}", f"↑ +{(sc['acc']-base['acc'])*100:.1f} PTS", "up")
card(c3, "leak broken", f"{sc['collapsed_acc']:.1%}", f"↓ {(sc['collapsed_acc']-sc['acc'])*100:.1f} PTS", "down")
card(c4, "sex reconstructed", f"{proxy['auc']:.2f}", "AUC · SEX REMOVED", "flat")

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── 1 ──────────────────────────────────────────────────────────────────
L, Rt = st.columns([0.42, 0.58], gap="large")
with L:
    st.markdown('<div class="section-num">01 / SHORTCUT DETECTION</div>', unsafe_allow_html=True)
    st.markdown("### One leaked feature inflates accuracy by 10 points")
    st.markdown(
        f'<div class="section-body">Planting a single feature that matches the label 95% of the '
        f'time lifts accuracy from {base["acc"]:.1%} to <b>{sc["acc"]:.1%}</b>. Permutation-reliance '
        f'ranking surfaces it without being told where to look. Permuting it then collapses the '
        f'model to <b>{sc["collapsed_acc"]:.1%}</b>, <i>below</i> the honest baseline.</div>',
        unsafe_allow_html=True)
with Rt:
    rel = dict(sorted(sc["reliance"].items(), key=lambda kv: kv[1], reverse=True)[:7])
    feats, vals = list(rel.keys()), list(rel.values())
    fig = go.Figure(go.Bar(
        y=feats[::-1], x=vals[::-1], orientation="h",
        marker=dict(color=[RED if f == sc["top_feature"] else GRAY for f in feats[::-1]], cornerradius=5),
        text=[f" {v:.3f}" for v in vals[::-1]], textposition="outside", textfont=dict(size=12, **MONO),
        hovertemplate="<b>%{y}</b><br>accuracy drop: %{x:.3f}<br>model leans on this feature<extra></extra>"))
    fig.update_layout(**PLOT, height=330,
                      xaxis=dict(title="accuracy drop when permuted", gridcolor="#1c2330",
                                 zeroline=False, range=[0, 0.245]),
                      yaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── 2 ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-num">02 / PROXY DETECTION</div>', unsafe_allow_html=True)
st.markdown("### Deleting the protected attribute doesn't delete it")
st.markdown(
    f'<div class="section-body">With <code>sex</code> dropped from the features, a probe model '
    f'rebuilds it at <b>{proxy["auc"]:.2f} AUC</b> from <code>{proxy["top_proxies"][0]}</code>, '
    f'<code>{proxy["top_proxies"][1]}</code>, and <code>{proxy["top_proxies"][2]}</code>. '
    f'Fairness-by-deletion is cosmetic when the remaining features still encode the attribute.</div>',
    unsafe_allow_html=True)
fig = go.Figure()
fig.add_shape(type="rect", x0=0.5, x1=1.0, y0=-0.16, y1=0.16, fillcolor="#1a2130", line_width=0)
fig.add_shape(type="rect", x0=0.5, x1=proxy["auc"], y0=-0.16, y1=0.16, fillcolor=RED, line_width=0)
fig.add_annotation(x=proxy["auc"] - 0.004, y=0, text=f"<b>{proxy['auc']:.2f}</b>", showarrow=False,
                   font=dict(size=17, color="#fff", **MONO), xanchor="right")
fig.add_annotation(x=0.5, y=-0.55, text="0.50 = unrecoverable", showarrow=False,
                   font=dict(size=12, color=MUTED), xanchor="left")
fig.add_annotation(x=1.0, y=-0.55, text="1.00 = fully encoded", showarrow=False,
                   font=dict(size=12, color=MUTED), xanchor="right")
fig.update_layout(**PLOT, height=170, xaxis=dict(range=[0.485, 1.015], visible=False),
                  yaxis=dict(range=[-0.9, 0.75], visible=False))
st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── 3 + 4 ──────────────────────────────────────────────────────────────
L, Rt = st.columns(2, gap="large")
with L:
    st.markdown('<div class="section-num">03 / SUBGROUP AUDIT</div>', unsafe_allow_html=True)
    st.markdown("### The average hides a 9-point gap")
    wg = R["worst_group"]["by_sex"]
    labels = [{"0": "female", "1": "male"}[k] for k in wg]
    vals = list(wg.values())
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    labels, vals = [labels[i] for i in order], [vals[i] for i in order]
    fig = go.Figure(go.Bar(
        x=labels, y=vals, marker=dict(color=[RED, GRAY], cornerradius=6), width=0.42,
        text=[f"{v:.1%}" for v in vals], textposition="outside", textfont=dict(size=14, **MONO),
        hovertemplate="<b>%{x}</b><br>accuracy: %{y:.1%}<extra></extra>"))
    fig.add_hline(y=base["acc"], line_dash="dot", line_color=MUTED, line_width=1.5,
                  annotation_text=f"headline {base['acc']:.1%}", annotation_font=dict(size=11.5, color=MUTED))
    fig.update_layout(**PLOT, height=380,
                      yaxis=dict(range=[0.75, 1.0], title="accuracy", gridcolor="#1c2330", zeroline=False),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with Rt:
    st.markdown('<div class="section-num">04 / PREDICTIVE MULTIPLICITY</div>', unsafe_allow_html=True)
    st.markdown("### Equal models, conflicting answers")
    band = st.slider(
        "disagreement band. a person counts as contested when between "
        "band and (1 − band) of the 25 models vote 'yes'",
        min_value=0.05, max_value=0.50, value=0.20, step=0.05)
    a_pct = substantial_pct(mp_a["frac_pos"], band)
    m_pct = substantial_pct(mp_m["frac_pos"], band)
    fig = go.Figure(go.Bar(
        x=["Adult<br>clean signal", "SPY next-day<br>weak signal"],
        y=[a_pct, m_pct],
        marker=dict(color=[GRAY, RED], cornerradius=6), width=0.42,
        text=[f"{a_pct:.1f}%", f"{m_pct:.1f}%"], textposition="outside",
        textfont=dict(size=15, **MONO),
        hovertemplate="<b>%{x}</b><br>contested individuals: %{y:.1f}%<br>"
                      f"band: {band:.2f}–{1-band:.2f}<extra></extra>"))
    fig.update_layout(**PLOT, height=305,
                      yaxis=dict(title="contested individuals", range=[0, 105], ticksuffix="%",
                                 gridcolor="#1c2330", zeroline=False),
                      xaxis=dict(showgrid=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown(
    f'<div class="section-body" style="max-width:100%; margin-top:1.2rem;">Twenty-five models, '
    f'distinguishable by nothing but the random seed and a 90% data subsample, agree on the '
    f'aggregate score, and give conflicting answers for the same person. Adult: 25 models at '
    f'{mp_a["acc_min"]:.1%}–{mp_a["acc_max"]:.1%}. SPY: {mp_m["acc_min"]:.1%}–{mp_m["acc_max"]:.1%} '
    f'(none beat always-up, {mp_m["up_rate"]:.0%}). Drag the slider: even at the strictest band, '
    f'the weak-signal disagreement doesn\'t go away.</div>', unsafe_allow_html=True)

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── 5 ──────────────────────────────────────────────────────────────────
L, Rt = st.columns([0.42, 0.58], gap="large")
with L:
    st.markdown('<div class="section-num">05 / TEMPORAL LEAKAGE</div>', unsafe_allow_html=True)
    st.markdown("### The backtest reads 66%. The honest number is 58%.")
    st.markdown(
        f'<div class="section-body">Same features, same model. With 20-day overlapping labels, a '
        f'shuffled backtest leaks near-duplicate days across the split and reads '
        f'<b>{leak["overlapping"]["random"]:.1%}</b>; walk-forward validation reads '
        f'<b>{leak["overlapping"]["walkforward"]:.1%}</b>. On clean 1-day labels the two agree. '
        f'the detector fires only when there is something to catch.</div>', unsafe_allow_html=True)
with Rt:
    cats = ["clean labels", "overlapping labels"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="random split (naive backtest)", x=cats,
                         y=[leak["clean"]["random"], leak["overlapping"]["random"]],
                         marker=dict(color=RED, cornerradius=5), width=0.3,
                         text=[f"{leak['clean']['random']:.1%}", f"{leak['overlapping']['random']:.1%}"],
                         textposition="outside", textfont=dict(size=13, **MONO),
                         hovertemplate="<b>random split</b><br>%{x}: %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(name="walk-forward (honest)", x=cats,
                         y=[leak["clean"]["walkforward"], leak["overlapping"]["walkforward"]],
                         marker=dict(color=GRAY, cornerradius=5), width=0.3,
                         text=[f"{leak['clean']['walkforward']:.1%}", f"{leak['overlapping']['walkforward']:.1%}"],
                         textposition="outside", textfont=dict(size=13, **MONO),
                         hovertemplate="<b>walk-forward</b><br>%{x}: %{y:.1%}<extra></extra>"))
    fig.update_layout(**PLOT, height=380, barmode="group", bargroupgap=0.1,
                      yaxis=dict(range=[0.42, 0.74], title="accuracy", gridcolor="#1c2330", zeroline=False),
                      xaxis=dict(showgrid=False),
                      legend=dict(orientation="h", y=1.15, x=0, font=dict(size=12)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── 6: COMPAS ──────────────────────────────────────────────────────────
st.markdown('<div class="section-num">06 / CASE STUDY: COMPAS</div>', unsafe_allow_html=True)
st.markdown("### The risk score behind real bail decisions is partly a coin flip")
cm = compas["multiplicity"]
c_band_pct = substantial_pct(cm["frac_pos"], 0.2)
st.markdown(
    f'<div class="section-body">A recidivism model trained on ProPublica\'s COMPAS data, '
    f'{compas["n"]:,} real defendants from Broward County, the dataset behind the landmark '
    f'2016 "Machine Bias" investigation, scores <b>{compas["acc"]:.1%}</b>. The audits: with '
    f'<code>race</code> removed, a probe rebuilds it at <b>{compas["proxy"]["auc"]:.2f} AUC</b> '
    f'from <code>{compas["proxy"]["top_proxies"][0]}</code> and '
    f'<code>{compas["proxy"]["top_proxies"][1]}</code>. And <b>{c_band_pct:.0f}% of defendants</b> '
    f'get conflicting risk calls from 25 equally-accurate models '
    f'({cm["acc_min"]:.1%}–{cm["acc_max"]:.1%}). For them, the flag depended on the random seed. '
    f'On a bail decision.</div>', unsafe_allow_html=True)

d1, d2, d3 = st.columns(3, gap="medium")
card(d1, "model accuracy", f"{compas['acc']:.1%}", f"{compas['n']:,} DEFENDANTS", "flat")
card(d2, "race reconstructed", f"{compas['proxy']['auc']:.2f}", "AUC · RACE REMOVED", "flat")
card(d3, "contested defendants", f"{c_band_pct:.0f}%", "ACROSS 25 EQUAL MODELS", "down")

st.markdown('<div class="spacer"></div>', unsafe_allow_html=True)

# ── CTA ────────────────────────────────────────────────────────────────
st.markdown("### Run these checks on your own model")
st.markdown(
    '<div class="section-body">Upload a trained sklearn-compatible classifier and a test CSV. '
    'Litmus runs the same audits live: reliance ranking, proxy detection, subgroup gaps, '
    'and multiplicity.</div>', unsafe_allow_html=True)
st.page_link("pages/1_Audit_your_model.py", label="Upload a model and audit it live →")

st.markdown(
    '<div class="footer"><a href="https://github.com/Anvita-Reddy/litmus">github.com/Anvita-Reddy/litmus</a>'
    ' &nbsp;·&nbsp; code · tests · CI &nbsp;·&nbsp; every number reproducible with one command</div>',
    unsafe_allow_html=True)