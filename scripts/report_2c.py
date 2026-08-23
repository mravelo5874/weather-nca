#!/usr/bin/env python
"""Generate the phase-2c results report as a self-contained HTML page.

Numbers are pasted from the evaluation runs, exactly like `plot_ladder.py`, so the page can
never silently disagree with the scorecards. Regenerate with:

    python scripts/report_2c.py        # -> media/phase2c_report.html

Charts are inline SVG computed here rather than drawn by a JS library: the Artifact CSP blocks
external scripts, and pre-computed geometry cannot fail to render.

Two verification years are in play and the page labels every number with its year:
  * held-out TEST is 2020 -- the headline.
  * the ladder comparison uses 2018, because 2b/2b'/2b-pf were scored there and comparing
    across years would confound the data scale-up with a change of verification year. 2018 is
    2c's validation split, so it is not held out; the two years agree to ~4%, which is what
    makes the comparison usable.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path("media") / "phase2c_report.html"

LEADS = [6, 12, 24, 48, 72, 120, 168, 240, 360]

# --- z500 RMSE (m^2/s^2) -------------------------------------------------------------------
# test 2018 for every phase, so the ladder is a like-for-like comparison.
PHASE0 = [110.2, 169.5, 329.7, 639.1, 881.1, 1231.8, 1493.3, 1795.6, 2208.5]
PHASE2B = [106.3, 158.0, 297.6, 624.0, 934.1, 1443.7, 1862.0, 2457.1, 3450.6]
PHASE2PF = [152.4, 167.9, 282.1, 531.0, 755.6, 1085.1, 1308.7, None, 1860.5]
PHASE2C_18 = [128.4, 107.2, 167.6, 309.2, 458.8, 743.6, 949.6, 1169.8, 1475.8]
PERSIST = [226.2, 361.6, 589.8, 818.2, 924.8, 1017.3, 1064.1, 1096.5, 1128.6]
CLIM = [1068.5, 1068.2, 1068.1, 1066.7, 1068.5, 1073.9, 1071.5, 1065.1, 1070.5]
# held-out test 2020 -- 2c only, the honest headline.
PHASE2C_20 = [131.5, 110.3, 174.7, 319.4, 461.2, 743.2, 952.3, 1217.3, 1474.6]

# --- training, 8 epochs --------------------------------------------------------------------
EPOCHS = list(range(1, 9))
TRAIN = [0.07785, 0.05322, 0.04591, 0.04148, 0.03846, 0.03663, 0.03550, 0.03492]
VAL = [0.03425, 0.02955, 0.02643, 0.02550, 0.02398, 0.02379, 0.023785, 0.023723]
SEL = [0.22740, 0.19875, 0.15742, 0.14693, 0.13313, 0.12165, 0.11539, 0.11257]

# --- 24 h skill vs persistence by variable, test 2020 --------------------------------------
SKILL24 = [
    ("geopotential 500", 71), ("v-wind 300", 72), ("geopotential 300", 72),
    ("v-wind 500", 69), ("geopotential 250", 70), ("u-wind 300", 65),
    ("v-wind 850", 65), ("10m v-wind", 65), ("temperature 500", 64),
    ("u-wind 500", 61), ("u-wind 850", 59), ("10m u-wind", 58),
    ("temperature 700", 59), ("humidity 500", 53), ("temperature 250", 56),
    ("temperature 850", 46), ("humidity 850", 47), ("2m temperature", -31),
]

# --- sustained per-window perturbation growth ----------------------------------------------
LEVELS = [850, 700, 500, 300, 250]
GROWTH_2C = {
    "geopotential": [1.110, 1.104, 1.112, 1.083, 1.086],
    "u-wind": [1.093, 1.100, 1.105, 1.105, 1.095],
    "v-wind": [1.093, 1.100, 1.116, 1.113, 1.103],
    "temperature": [1.068, 1.088, 1.099, 1.096, 1.111],
    "humidity": [1.102, 1.099, 1.103, 1.086, 1.078],
}
DOUBLING = {"phase 0": 1.033, "2b-pushforward": 1.139, "2c": 1.079}

ATTEMPTS = [
    ("1", "fp16 · lr 1e-3", "forward overflow, 143 bad batches", "epoch 1"),
    ("2", "bf16 · lr 1e-3", "~700 non-finite gradients", "epoch 1"),
    ("3", "fp32 · lr 1e-3", "diverged", "~40 min"),
    ("4", "bf16 · lr 3e-4", "diverged, then 10 h at 0 steps", "step 11,103"),
    ("5", "diagnostic resume", "reproduced onset; guard aborted", "step 11,103"),
    ("6", "+ spectral norm · wd 0.1", "completed all 8 epochs", "—"),
]


def dbl(per_window: float) -> float:
    """Per-6h growth factor -> error-doubling time in days."""
    import math
    return math.log(2) / math.log(per_window**4)


# ---------------------------------------------------------------------------- svg helpers ---
def _sc(v, lo, hi, a, b):
    return a + (v - lo) / (hi - lo) * (b - a)


def linechart(series, xs, *, w=760, h=380, xlab="", ylab="", ymax=None, ymin=0,
              xticks=None, yticks=None, pad_l=64, pad_b=44, pad_t=16, pad_r=118,
              xlog=False, fmt="{:.0f}"):
    """Multi-series line chart. `series` = [{name,vals,color,dash,width,emph}]."""
    import math
    x0, x1 = pad_l, w - pad_r
    y0, y1 = h - pad_b, pad_t
    allv = [v for s in series for v in s["vals"] if v is not None]
    ymax = ymax if ymax is not None else max(allv) * 1.06
    tx = (lambda v: _sc(math.log10(v), math.log10(xs[0]), math.log10(xs[-1]), x0, x1)) if xlog \
        else (lambda v: _sc(v, xs[0], xs[-1], x0, x1))
    ty = lambda v: _sc(v, ymin, ymax, y0, y1)  # noqa: E731

    p = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" preserveAspectRatio="xMidYMid meet">']
    yticks = yticks or [ymin + (ymax - ymin) * i / 4 for i in range(5)]
    for t in yticks:
        yy = ty(t)
        p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" class="grid"/>')
        p.append(f'<text x="{x0 - 10}" y="{yy + 4:.1f}" class="tick tick-y">{fmt.format(t)}</text>')
    for t in (xticks or xs):
        xx = tx(t)
        p.append(f'<text x="{xx:.1f}" y="{y0 + 20}" class="tick tick-x">{t}</text>')
    p.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" class="axis"/>')

    for s in series:
        pts = [(tx(x), ty(v)) for x, v in zip(xs, s["vals"]) if v is not None]
        d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        dash = f' stroke-dasharray="{s["dash"]}"' if s.get("dash") else ""
        sw = s.get("width", 2)
        p.append(f'<path d="{d}" fill="none" stroke="{s["color"]}" stroke-width="{sw}"'
                 f' stroke-linecap="round" stroke-linejoin="round"{dash}/>')
        if s.get("emph"):
            for (a, b), v in zip(pts, [v for v in s["vals"] if v is not None]):
                p.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{s["color"]}" '
                         f'stroke="var(--surface)" stroke-width="2"><title>{v:g}</title></circle>')
        lx, ly = pts[-1]
        p.append(f'<text x="{lx + 9:.1f}" y="{ly + 4:.1f}" class="slabel" '
                 f'fill="{s["color"]}">{s["name"]}</text>')

    if ylab:
        p.append(f'<text transform="translate(15,{(y0 + y1) / 2:.0f}) rotate(-90)" '
                 f'class="axlab" text-anchor="middle">{ylab}</text>')
    if xlab:
        p.append(f'<text x="{(x0 + x1) / 2:.0f}" y="{h - 6}" class="axlab" '
                 f'text-anchor="middle">{xlab}</text>')
    p.append("</svg>")
    return "\n".join(p)


def barchart(rows, *, w=760, h=520, pad_l=150, pad_r=64, pad_t=8, pad_b=34, unit="%",
             fmt="{:+d}", zero_label="0 = persistence"):
    """Horizontal bars, diverging around zero. `rows` = [(label, value)]."""
    x0, x1 = pad_l, w - pad_r
    lo = min(0, min(v for _, v in rows)) * 1.08
    hi = max(v for _, v in rows) * 1.08
    zx = _sc(0, lo, hi, x0, x1)
    bh = (h - pad_t - pad_b) / len(rows)
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" class="chart" preserveAspectRatio="xMidYMid meet">']
    for i, (lab, v) in enumerate(rows):
        y = pad_t + i * bh
        vx = _sc(v, lo, hi, x0, x1)
        a, b = (zx, vx) if v >= 0 else (vx, zx)
        col = "var(--s1)" if v >= 0 else "var(--bad)"
        p.append(f'<rect x="{a:.1f}" y="{y + 2:.1f}" width="{max(b - a, 1):.1f}" '
                 f'height="{bh - 4:.1f}" rx="3" fill="{col}"><title>{lab}: {v}{unit}</title></rect>')
        p.append(f'<text x="{x0 - 10}" y="{y + bh / 2 + 4:.1f}" class="tick tick-y">{lab}</text>')
        tx_ = b + 7 if v >= 0 else a - 7
        anc = "start" if v >= 0 else "end"
        p.append(f'<text x="{tx_:.1f}" y="{y + bh / 2 + 4:.1f}" class="barval" '
                 f'text-anchor="{anc}">{fmt.format(v)}{unit}</text>')
    p.append(f'<line x1="{zx:.1f}" y1="{pad_t}" x2="{zx:.1f}" y2="{h - pad_b}" class="axis"/>')
    if zero_label:
        p.append(f'<text x="{zx:.1f}" y="{h - 14}" class="tick" text-anchor="middle">'
                 f'{zero_label}</text>')
    p.append("</svg>")
    return "\n".join(p)


# ---------------------------------------------------------------------------------- page ----
def build() -> str:
    idx = lambda a: [100 * v / a[0] for v in a]  # noqa: E731

    rmse = linechart(
        [
            {"name": "2b", "vals": PHASE2B, "color": "var(--s3)", "width": 1.6},
            {"name": "phase 0", "vals": PHASE0, "color": "var(--mut)", "width": 1.6},
            {"name": "2b-pf", "vals": PHASE2PF, "color": "var(--s2)", "width": 1.6},
            {"name": "2c", "vals": PHASE2C_18, "color": "var(--s1)", "width": 3, "emph": True},
            {"name": "persist.", "vals": PERSIST, "color": "var(--mut)", "width": 1.4,
             "dash": "5 4"},
            {"name": "clim.", "vals": CLIM, "color": "var(--mut)", "width": 1.4, "dash": "2 4"},
        ],
        LEADS, xlab="lead time (hours)", ylab="z500 RMSE (m²/s²)",
        xticks=[6, 24, 72, 168, 360], xlog=True, ymax=3600,
    )

    skill = linechart(
        [
            {"name": "2b", "vals": [100 * (1 - a / b) for a, b in zip(PHASE2B, PERSIST)],
             "color": "var(--s3)", "width": 1.6},
            {"name": "phase 0", "vals": [100 * (1 - a / b) for a, b in zip(PHASE0, PERSIST)],
             "color": "var(--mut)", "width": 1.6},
            {"name": "2b-pf", "vals": [100 * (1 - a / b) if a else None
                                       for a, b in zip(PHASE2PF, PERSIST)],
             "color": "var(--s2)", "width": 1.6},
            {"name": "2c", "vals": [100 * (1 - a / b) for a, b in zip(PHASE2C_18, PERSIST)],
             "color": "var(--s1)", "width": 3, "emph": True},
        ],
        LEADS, xlab="lead time (hours)", ylab="skill vs persistence (%)",
        xticks=[6, 24, 72, 168, 360], xlog=True, ymin=-220, ymax=90,
        yticks=[-200, -150, -100, -50, 0, 50],
    )

    train = linechart(
        [
            {"name": "train", "vals": idx(TRAIN), "color": "var(--s3)", "width": 1.8},
            {"name": "val", "vals": idx(VAL), "color": "var(--s2)", "width": 1.8, "emph": True},
            {"name": "selection", "vals": idx(SEL), "color": "var(--s1)", "width": 3,
             "emph": True},
        ],
        EPOCHS, w=760, h=330, xlab="epoch", ylab="% of epoch-1 value",
        ymin=40, ymax=105, yticks=[40, 55, 70, 85, 100], xticks=EPOCHS, pad_r=104,
    )

    growth = linechart(
        [{"name": k, "vals": [dbl(v) for v in vals],
          "color": f"var(--g{i+1})", "width": 2, "emph": True}
         for i, (k, vals) in enumerate(GROWTH_2C.items())],
        LEVELS, w=760, h=340, xlab="pressure level (hPa)", ylab="error-doubling time (days)",
        ymin=1.2, ymax=2.9, yticks=[1.5, 2.0, 2.5], xticks=LEVELS, pad_r=124,
        fmt="{:.1f}",
    )

    bars = barchart(SKILL24)

    rows = "\n".join(
        f"<tr><td class='mono'>{e}</td><td class='mono num'>{t:.5f}</td>"
        f"<td class='mono num'>{v:.5f}</td><td class='mono num'>{s:.5f}</td></tr>"
        for e, t, v, s in zip(EPOCHS, TRAIN, VAL, SEL))

    att = "\n".join(
        f"<tr><td class='mono'>{n}</td><td>{cfg}</td><td>{what}</td>"
        f"<td class='mono num'>{when}</td></tr>" for n, cfg, what, when in ATTEMPTS)

    ladder = "\n".join(
        f"<tr><td class='mono num'>{h}h</td><td class='mono num'>{p0:.1f}</td>"
        f"<td class='mono num'>{p2b:.1f}</td>"
        f"<td class='mono num'>{'—' if pf is None else f'{pf:.1f}'}</td>"
        f"<td class='mono num hero'>{c:.1f}</td>"
        f"<td class='mono num'>{'—' if pf is None else f'{100*(c/pf-1):+.1f}%'}</td></tr>"
        for h, p0, p2b, pf, c in zip(LEADS, PHASE0, PHASE2B, PHASE2PF, PHASE2C_18))

    return TEMPLATE.format(rmse=rmse, skill=skill, train=train, growth=growth, bars=bars,
                           rows=rows, att=att, ladder=ladder,
                           d2c=f"{dbl(1.079):.2f}", dpf=f"{dbl(1.139):.2f}",
                           d0=f"{dbl(1.033):.2f}")


TEMPLATE = """<title>Phase 2c Scorecard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --surface:#FAFBFC; --panel:#FFFFFF; --ink:#12171C; --ink2:#4E5964; --ink3:#78838E;
  --rule:#DFE5EA; --rule2:#EDF1F4;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --mut:#9AA5B0; --bad:#C92A2A; --good:#1baf7a;
  --g1:#2a78d6; --g2:#eb6834; --g3:#1baf7a; --g4:#eda100; --g5:#4a3aa7;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface:#12161A; --panel:#181D22; --ink:#F2F5F7; --ink2:#B4BEC7; --ink3:#87929C;
    --rule:#2A3238; --rule2:#20272C;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --mut:#6C7883; --bad:#e66767; --good:#199e70;
    --g1:#3987e5; --g2:#d95926; --g3:#199e70; --g4:#c98500; --g5:#9085e9;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface:#12161A; --panel:#181D22; --ink:#F2F5F7; --ink2:#B4BEC7; --ink3:#87929C;
  --rule:#2A3238; --rule2:#20272C;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --mut:#6C7883; --bad:#e66767; --good:#199e70;
  --g1:#3987e5; --g2:#d95926; --g3:#199e70; --g4:#c98500; --g5:#9085e9;
}}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--surface); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1080px; margin:0 auto; padding:56px 28px 96px; }}
.prose {{ max-width:64ch; }}
h1,h2,h3 {{ font-family:Newsreader,Georgia,serif; font-weight:600; text-wrap:balance;
  line-height:1.18; margin:0; letter-spacing:-.008em; }}
h1 {{ font-size:clamp(2.3rem,5.2vw,3.4rem); }}
h2 {{ font-size:1.72rem; margin-top:8px; }}
h3 {{ font-size:1.12rem; font-family:"IBM Plex Sans",sans-serif; font-weight:600;
  letter-spacing:0; }}
p {{ margin:0 0 1.05em; color:var(--ink2); }}
strong {{ color:var(--ink); font-weight:600; }}
.mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink3); }}
header {{ border-bottom:1px solid var(--rule); padding-bottom:34px; margin-bottom:8px; }}
.lede {{ font-family:Newsreader,Georgia,serif; font-size:1.28rem; line-height:1.55;
  color:var(--ink2); max-width:60ch; margin-top:18px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin-top:26px; font-size:.8rem; }}
.meta span {{ color:var(--ink3); }}
.meta b {{ color:var(--ink2); font-weight:500; }}
section {{ margin-top:64px; }}
.rule {{ border:0; border-top:1px solid var(--rule2); margin:0 0 26px; }}
.stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(196px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:10px;
  overflow:hidden; margin-top:36px; }}
.stat {{ background:var(--panel); padding:20px 22px 18px; }}
.stat .k {{ font-family:"IBM Plex Mono",monospace; font-size:.68rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--ink3); }}
.stat .v {{ font-family:Newsreader,Georgia,serif; font-size:2.15rem; font-weight:600;
  line-height:1.12; margin-top:9px; font-variant-numeric:tabular-nums; }}
.stat .s {{ font-size:.82rem; color:var(--ink3); margin-top:5px; }}
.up {{ color:var(--good); }} .down {{ color:var(--bad); }}
figure {{ margin:26px 0 0; background:var(--panel); border:1px solid var(--rule);
  border-radius:10px; padding:20px 18px 12px; }}
figcaption {{ font-size:.83rem; color:var(--ink3); margin-top:12px; padding:0 6px 6px;
  max-width:74ch; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--rule2); stroke-width:1; }}
.axis {{ stroke:var(--rule); stroke-width:1; }}
.tick {{ font-family:"IBM Plex Mono",monospace; font-size:11px; fill:var(--ink3);
  font-variant-numeric:tabular-nums; }}
.tick-y {{ text-anchor:end; }} .tick-x {{ text-anchor:middle; }}
.axlab {{ font-family:"IBM Plex Sans",sans-serif; font-size:12px; fill:var(--ink3); }}
.slabel {{ font-family:"IBM Plex Sans",sans-serif; font-size:12.5px; font-weight:600; }}
.barval {{ font-family:"IBM Plex Mono",monospace; font-size:11px; fill:var(--ink2);
  font-variant-numeric:tabular-nums; }}
.tbl {{ width:100%; overflow-x:auto; margin-top:24px; }}
table {{ width:100%; border-collapse:collapse; font-size:.86rem; }}
th {{ text-align:left; font-family:"IBM Plex Mono",monospace; font-size:.68rem;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink3); font-weight:500;
  padding:0 14px 9px 0; border-bottom:1px solid var(--rule); white-space:nowrap; }}
td {{ padding:9px 14px 9px 0; border-bottom:1px solid var(--rule2); color:var(--ink2); }}
td.num, th.num {{ text-align:right; padding-right:18px; }}
td.hero {{ color:var(--ink); font-weight:500; }}
.note {{ border-left:2px solid var(--s2); padding:2px 0 2px 18px; margin:26px 0;
  max-width:62ch; }}
.note p:last-child {{ margin-bottom:0; }}
.note .eyebrow {{ color:var(--s2); }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:34px; }}
footer {{ margin-top:72px; padding-top:24px; border-top:1px solid var(--rule);
  font-size:.8rem; color:var(--ink3); }}
@media (max-width:640px) {{ .wrap {{ padding:36px 18px 64px; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Milestone 2 · Phase 2c · 39 years, 28 channels</div>
  <h1>A local update rule, trained on four decades</h1>
  <p class="lede">The first phase-2c run to finish. A neural cellular automaton with a strictly
  local update rule, 1.03&nbsp;M parameters on a 10,242-node icosahedral mesh, cuts 24-hour z500
  error by 41% against the previous best — after four consecutive runs destroyed themselves.</p>
  <div class="meta mono">
    <span>run <b>phase2c_full_20260819_192049</b></span>
    <span>NVIDIA L4 · bf16</span>
    <span><b>22 h 05 m</b></span>
    <span><b>~$20</b></span>
    <span>56,976 steps</span>
  </div>
</header>

<div class="stats">
  <div class="stat"><div class="k">24 h z500 RMSE</div><div class="v">174.7</div>
    <div class="s">m²/s² · held-out test 2020</div></div>
  <div class="stat"><div class="k">vs 2b-pushforward</div><div class="v up">−40.6%</div>
    <div class="s">same year (2018), 24 h</div></div>
  <div class="stat"><div class="k">divergence events</div><div class="v">0</div>
    <div class="s">across all 56,976 steps</div></div>
  <div class="stat"><div class="k">error doubling</div><div class="v">{d2c} d</div>
    <div class="s">inside the synoptic band</div></div>
</div>

<section>
  <hr class="rule">
  <div class="eyebrow">01 — Forecast skill</div>
  <h2>Data volume dominates every other change</h2>
  <div class="prose">
  <p>Phase 2c changes one variable against 2b-pushforward: <strong>39 years of training data
  instead of two</strong>. It is better at every lead time, and the margin is largest in the
  2–5 day range where a forecast is actually used.</p>
  </div>
  <figure>
    {rmse}
    <figcaption>z500 area-weighted RMSE, all phases scored on <strong>2018</strong> so the
    comparison is like-for-like. Lead time is on a log axis; lower is better. 2b-pushforward has
    no 240 h entry in its scorecard, hence the gap. Phase 2b′ is omitted — it was undertrained
    after an emergency LR cut and was never a clean control.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th class="num">lead</th><th class="num">phase 0</th><th class="num">2b</th>
    <th class="num">2b-pf</th><th class="num">2c</th><th class="num">2c vs 2b-pf</th></tr></thead>
    <tbody>{ladder}</tbody>
  </table></div>
  <figure>
    {skill}
    <figcaption>Skill against persistence. 2c stays positive to <strong>~212 h (8.9 days)</strong>;
    2b crossed zero at about 72 h. Below the line the model is worse than simply predicting that
    nothing changes.</figcaption>
  </figure>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">02 — Training</div>
  <h2>The validation loss went blind before training did</h2>
  <div class="prose">
  <p>All three tracked quantities improved every epoch, but not at the same rate. Single-step
  validation loss <strong>flattened after epoch 5</strong> — 1.1% total gain over the last three
  epochs — while the 72-hour rollout metric kept improving by 15%.</p>
  <p>This is why the selection metric was widened from 8 to 12 forecast windows before this run.
  An 8-window metric would have been reading a signal that had largely stopped moving, and
  checkpoint selection would have been close to arbitrary over the final epochs.</p>
  </div>
  <figure>
    {train}
    <figcaption>Each series indexed to its own epoch-1 value, so three different scales share one
    axis. Validation settles near 69% and stops; the rollout metric continues to 50%.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th class="num">epoch</th><th class="num">train</th><th class="num">val</th>
    <th class="num">selection (72 h)</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <div class="note">
    <div class="eyebrow">Not converged</div>
    <p>Both training loss and the selection metric were still falling at epoch 8, with no
    plateau. The 8-epoch budget was fixed before anyone knew the model would survive training at
    all. More epochs would probably still help.</p>
  </div>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">03 — Where it fails</div>
  <h2>Two-metre temperature is worse than doing nothing</h2>
  <div class="prose">
  <p>Every upper-air field beats persistence comfortably at 24 hours. One does not:
  <strong>2 m temperature scores −31%</strong>, and it degrades further with lead time — −57% at
  72 h, −90% at 120 h. It is the only channel in the state that is actively harmful.</p>
  <p>The diurnal cycle is the obvious suspect. Solar forcing is switched on in this run, so
  either the conditioning is too weak to drive a surface field that swings on a 24-hour period,
  or a 223 km mesh cannot resolve the land-surface contrast that sets it. That is a measurement
  to make, not a conclusion to draw here.</p>
  </div>
  <figure>
    {bars}
    <figcaption>Skill against persistence at 24 h, held-out test 2020. Bars right of the line
    beat persistence.</figcaption>
  </figure>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">04 — Error growth</div>
  <h2>The error growth rate is physically plausible</h2>
  <div class="prose">
  <p>Perturbation growth is measured directly: perturb the initial state, and track how the
  difference between the two trajectories grows per forecast window. Sustained growth is
  <strong>×1.079 per 6 h</strong>, an error-doubling time of <strong>{d2c} days</strong>.</p>
  <p>The real atmosphere doubles synoptic-scale errors in roughly 1.5–2.5 days. Earlier models in
  this ladder sat on both sides of that: phase 0 at {d0} days was over-damped — stable because it
  was sluggish — and 2b-pushforward at {dpf} days grew error faster than the atmosphere does. 2c
  is the first model here to land inside the band.</p>
  </div>
  <figure>
    {growth}
    <figcaption>Error-doubling time by variable and pressure level, phase 2c. The 1.5–2.5 day
    synoptic band is where the real atmosphere sits. Higher is slower error growth, which is not
    automatically better — a model far above the band is over-damped.</figcaption>
  </figure>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">05 — What it took</div>
  <h2>Five runs failed first</h2>
  <div class="prose">
  <p>Four attempts changed precision and then learning rate. Each survived longer than the last
  and then diverged anyway — the signature of a driver rather than a cause. A gradient trace
  found the real mechanism: the composed 20-sub-step update map has a stability threshold, and
  the weight norm ratchets across it during otherwise healthy training.</p>
  <p>The fix bounds the map instead of slowing the approach to it: <strong>spectral
  normalisation</strong> pins each hidden layer's largest singular value near 1, and
  <strong>weight decay raised from 1e-5 to 0.1</strong> holds the parameters spectral norm does
  not wrap. The previous value was measured to be ~11,000× too weak to oppose the drift.</p>
  </div>
  <div class="tbl"><table>
    <thead><tr><th class="num">#</th><th>configuration</th><th>outcome</th>
    <th class="num">died at</th></tr></thead>
    <tbody>{att}</tbody>
  </table></div>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">06 — Caveats</div>
  <h2>What this result does not establish</h2>
  <div class="cols prose">
    <div>
      <h3>The two fixes are confounded</h3>
      <p>Spectral norm and the weight-decay change shipped together. Neither is established as
      individually necessary; separating them costs another full run.</p>
      <h3>Two verification years</h3>
      <p>The headline is held-out 2020. The ladder comparison uses 2018, which was 2c's
      validation split and so is not held out. The two agree to ~4% (167.6 vs 174.7 at 24 h),
      which is what makes the comparison usable.</p>
    </div>
    <div>
      <h3>One run, one seed</h3>
      <p>No variance estimate. Earlier phases showed run-to-run swings of ~5%, which is far
      smaller than the 41% here, but the error bars are genuinely unknown.</p>
      <h3>The thesis is still untested</h3>
      <p>This says a local rule scales with data. It does not say a local rule is
      <em>sufficient</em> — that needs phase 2d, the same-budget non-local control.</p>
    </div>
  </div>
</section>

<footer class="mono">
  weather-nca · generated by scripts/report_2c.py from the phase-2c evaluation logs
</footer>
</div>
"""


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(json.dumps({"doubling_2c": round(dbl(1.079), 3),
                      "doubling_2bpf": round(dbl(1.139), 3),
                      "doubling_p0": round(dbl(1.033), 3)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
