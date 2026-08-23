#!/usr/bin/env python
"""Generate the phase-2d locality-control report as a self-contained HTML page.

Numbers are pasted from the evaluation runs (scripts/eval logs under ~/evalout on the cloud
instance), exactly like `report_2c.py`, so the page can never silently disagree with the
scorecards. Chart helpers are imported from `report_2c` rather than duplicated.

    python scripts/report_2d.py        # -> media/phase2d_report.html

Covers the four arms that have finished. The fifth -- the parameter-matched `d=1` placebo --
is still training, and it is the one that carries the pre-registered comparison, so this page
reports what is known and marks what is not.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_2c import barchart, dbl, linechart  # noqa: E402

OUT = Path("media") / "phase2d_report.html"

LEADS = [6, 12, 24, 48, 72, 120, 168, 240, 360]

# z500 area-weighted RMSE (m^2/s^2), HELD-OUT TEST 2020, identical eval settings.
A = [131.5, 110.3, 174.7, 319.4, 461.2, 743.2, 952.3, 1217.3, 1474.6]   # 2c seed 0
B = [129.0, 110.2, 173.3, 318.2, 460.6, 734.3, 944.4, 1133.6, 1305.5]   # 2c seed 1
C = [232.6, 137.9, 291.2, 711.1, 1014.0, 1390.4, 1594.9, 1746.9, 1983.5]  # control GNN
D = [129.9, 108.5, 173.4, 320.2, 467.0, 763.2, 977.9, 1228.2, 1556.7]   # dilated d=8
PERSIST = [229.6, 364.7, 593.5, 829.7, 932.9, 1043.1, 1102.8, 1134.7, 1138.1]
CLIM = [1105.8, 1106.2, 1104.6, 1102.7, 1099.3, 1092.4, 1099.6, 1102.0, 1097.1]

# Sustained per-window perturbation growth (the direct diagnostic).
GROWTH = {"A · local, seed 0": 1.079, "B · local, seed 1": 1.090,
          "D · dilated d=8": 1.097, "C · control GNN": 1.195}

# 2 m temperature skill vs persistence at every 6 h lead, arm A. Starred leads are multiples
# of 24 h, where persistence is diurnally aligned and therefore an unusually strong baseline.
T2M_LEADS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
T2M = [36, 51, 28, -31, 15, 24, 5, -42, -4, 8, -12, -57]
Z500_SKILL = [43, 70, 66, 71, 66, 66, 63, 62, 58, 56, 53, 51]

ARMS = [
    ("A", "phase 2c, seed 0", "local NCA, 20 sub-steps &times; 1 hop",
     "4 &mdash; id, &nabla;x, &nabla;y, &nabla;&sup2;", "20 hops", "1,029,692",
     "the reference model &mdash; strictly local"),
    ("B", "phase 2c, seed 1", "identical to A", "4", "20 hops", "1,029,692",
     "<b>only the seed differs from A</b> &mdash; this pair is the noise floor"),
    ("C", "phase 2d control", "message-passing GNN, <b>1 pass</b>, 6 hops",
     "n/a", "<b>9 hops</b> (measured)", "1,010,780",
     "non-local <i>topology</i> but less reach; also differs in iteration count, "
     "weight sharing and featurisation"),
    ("D", "phase 2d dilated", "local NCA <b>+ ring at 8 hops</b>",
     "5 &mdash; id, &nabla;x, &nabla;y, &nabla;&sup2;, ring<sub>8</sub>",
     "<b>160 hops</b> &asymp; 1.7&times; mesh diameter", "1,060,412",
     "<b>the treatment</b> &mdash; identical to A except perception sees globally"),
    ("E", "phase 2d dilated, d=1", "local NCA + ring at 1 hop",
     "5 &mdash; &hellip; ring<sub>1</sub>", "20 hops (= A)", "1,060,412 (= D)",
     "<b>the placebo</b> &mdash; parameter- and compute-identical to D, no extra reach. "
     "<b>Still training.</b>"),
]


def build() -> str:
    rmse = linechart(
        [
            {"name": "C · GNN", "vals": C, "color": "var(--s2)", "width": 2},
            {"name": "A · local", "vals": A, "color": "var(--s1)", "width": 2.6,
             "emph": True},
            {"name": "B · local s1", "vals": B, "color": "var(--s1)", "width": 1.8,
             "dash": "6 4"},
            {"name": "D · d=8", "vals": D, "color": "var(--s3)", "width": 2.6, "emph": True},
            {"name": "persist.", "vals": PERSIST, "color": "var(--mut)", "width": 1.4,
             "dash": "5 4"},
            {"name": "clim.", "vals": CLIM, "color": "var(--mut)", "width": 1.4, "dash": "2 4"},
        ],
        LEADS, xlab="lead time (hours)", ylab="z500 RMSE (m²/s²)",
        xticks=[6, 24, 72, 168, 360], xlog=True, ymax=2100,
    )

    spread = [(f"{h} h", round(100 * abs(a - b) / ((a + b) / 2), 1))
              for h, a, b in zip(LEADS, A, B)]
    spread_chart = barchart(spread, h=330, pad_l=72, unit="%", fmt="{:.1f}",
                            zero_label="")

    growth_rows = [(k, round(dbl(v), 2)) for k, v in
                   sorted(GROWTH.items(), key=lambda kv: -dbl(kv[1]))]
    growth_chart = barchart(growth_rows, h=250, pad_l=170, unit=" d",
                            fmt="{:.2f}", zero_label="")

    t2m = linechart(
        [
            {"name": "z500", "vals": Z500_SKILL, "color": "var(--s1)", "width": 2},
            {"name": "2 m temp", "vals": T2M, "color": "var(--s2)", "width": 2.6,
             "emph": True},
        ],
        T2M_LEADS, w=760, h=330, xlab="lead time (hours)",
        ylab="skill vs persistence (%)", ymin=-70, ymax=80,
        yticks=[-60, -30, 0, 30, 60], xticks=T2M_LEADS, pad_r=104,
    )

    ladder = "\n".join(
        f"<tr><td class='mono num'>{h}h</td><td class='mono num'>{a:.1f}</td>"
        f"<td class='mono num'>{b:.1f}</td>"
        f"<td class='mono num' style='color:var(--ink3)'>{100*abs(a-b)/((a+b)/2):.1f}%</td>"
        f"<td class='mono num hero'>{d:.1f}</td>"
        f"<td class='mono num'>{100*(d/((a+b)/2)-1):+.1f}%</td>"
        f"<td class='mono num'>{c:.1f}</td></tr>"
        for h, a, b, c, d in zip(LEADS, A, B, C, D))

    arms = "\n".join(
        f"<tr><td class='mono hero'>{t}</td><td class='mono'>{cfg}</td><td>{rule}</td>"
        f"<td>{grp}</td><td class='mono'>{reach}</td><td class='mono num'>{par}</td>"
        f"<td>{diff}</td></tr>"
        for t, cfg, rule, grp, reach, par, diff in ARMS)

    return TEMPLATE.format(rmse=rmse, spread=spread_chart, growth=growth_chart, t2m=t2m,
                           ladder=ladder, arms=arms,
                           dA=f"{dbl(1.079):.2f}", dB=f"{dbl(1.090):.2f}",
                           dC=f"{dbl(1.195):.2f}", dD=f"{dbl(1.097):.2f}")


TEMPLATE = """<title>Phase 2d Locality Control</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --surface:#FAFBFC; --panel:#FFFFFF; --ink:#12171C; --ink2:#4E5964; --ink3:#78838E;
  --rule:#DFE5EA; --rule2:#EDF1F4;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --mut:#9AA5B0; --bad:#C92A2A; --good:#1baf7a;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --surface:#12161A; --panel:#181D22; --ink:#F2F5F7; --ink2:#B4BEC7; --ink3:#87929C;
    --rule:#2A3238; --rule2:#20272C;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --mut:#6C7883; --bad:#e66767; --good:#199e70;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --surface:#12161A; --panel:#181D22; --ink:#F2F5F7; --ink2:#B4BEC7; --ink3:#87929C;
  --rule:#2A3238; --rule2:#20272C;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --mut:#6C7883; --bad:#e66767; --good:#199e70;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif; font-size:16px; line-height:1.62;
  -webkit-font-smoothing:antialiased; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:56px 28px 96px; }}
.prose {{ max-width:64ch; }}
h1,h2 {{ font-family:Newsreader,Georgia,serif; font-weight:600; text-wrap:balance;
  line-height:1.18; margin:0; letter-spacing:-.008em; }}
h1 {{ font-size:clamp(2.3rem,5.2vw,3.4rem); }}
h2 {{ font-size:1.72rem; margin-top:8px; }}
h3 {{ font-size:1.1rem; font-weight:600; margin:0 0 6px; }}
p {{ margin:0 0 1.05em; color:var(--ink2); }}
strong {{ color:var(--ink); font-weight:600; }}
.mono {{ font-family:"IBM Plex Mono",ui-monospace,monospace; font-variant-numeric:tabular-nums; }}
.eyebrow {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink3); }}
header {{ border-bottom:1px solid var(--rule); padding-bottom:34px; margin-bottom:8px; }}
.lede {{ font-family:Newsreader,Georgia,serif; font-size:1.28rem; line-height:1.55;
  color:var(--ink2); max-width:60ch; margin-top:18px; }}
.meta {{ display:flex; flex-wrap:wrap; gap:10px 26px; margin-top:26px; font-size:.8rem; }}
.meta span {{ color:var(--ink3); }} .meta b {{ color:var(--ink2); font-weight:500; }}
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
  padding:0 14px 9px 0; border-bottom:1px solid var(--rule); white-space:nowrap;
  vertical-align:bottom; }}
td {{ padding:9px 14px 9px 0; border-bottom:1px solid var(--rule2); color:var(--ink2);
  vertical-align:top; }}
td.num, th.num {{ text-align:right; padding-right:18px; }}
td.hero {{ color:var(--ink); font-weight:600; }}
.note {{ border-left:2px solid var(--s2); padding:2px 0 2px 18px; margin:26px 0;
  max-width:64ch; }}
.note p:last-child {{ margin-bottom:0; }} .note .eyebrow {{ color:var(--s2); }}
.note.open {{ border-color:var(--s1); }} .note.open .eyebrow {{ color:var(--s1); }}
.cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:34px; }}
footer {{ margin-top:72px; padding-top:24px; border-top:1px solid var(--rule);
  font-size:.8rem; color:var(--ink3); }}
@media (max-width:640px) {{ .wrap {{ padding:36px 18px 64px; }} }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Milestone 2 · Phase 2d · the thesis under test</div>
  <h1>Does a local rule need to see further?</h1>
  <p class="lede">Give a strictly local weather model global reach &mdash; 160 mesh hops per
  6-hour window, 1.7&times; the diameter of the planet's grid &mdash; and its 24-hour forecast
  error moves by less than the gap between two random seeds.</p>
  <div class="meta mono">
    <span>held-out test <b>2020</b></span>
    <span><b>4 of 5</b> arms evaluated</span>
    <span>NVIDIA L4 · bf16</span>
    <span>39 years · 28 channels · 10,242 nodes</span>
  </div>
</header>

<div class="stats">
  <div class="stat"><div class="k">24 h z500 · local (A/B)</div><div class="v">174.7 / 173.3</div>
    <div class="s">m²/s² · the two seeds</div></div>
  <div class="stat"><div class="k">24 h z500 · global reach</div><div class="v">173.4</div>
    <div class="s">inside the seed spread</div></div>
  <div class="stat"><div class="k">seed spread at 360 h</div><div class="v down">12.2%</div>
    <div class="s">vs 0.8% at 24 h</div></div>
  <div class="stat"><div class="k">control GNN, 24 h</div><div class="v down">+67%</div>
    <div class="s">291.2 m²/s²</div></div>
</div>

<section>
  <hr class="rule">
  <div class="eyebrow">The arms</div>
  <h2>What is being compared, and what differs</h2>
  <div class="tbl"><table>
    <thead><tr><th></th><th>config</th><th>update rule</th><th>perception</th>
    <th>reach / window</th><th class="num">params</th>
    <th>the one thing that makes it different</th></tr></thead>
    <tbody>{arms}</tbody>
  </table></div>
  <div class="note open">
    <div class="eyebrow">The pre-registered comparison is D vs E, and E is unfinished</div>
    <p>D and E are identical in parameters and wall-clock and differ only in reach, so their
    difference is attributable to reach alone. D against A is <em>not</em> that comparison: A
    has 3% fewer parameters, and the sign of that confound is unknown. Everything below is
    therefore evidence, not the verdict.</p>
  </div>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">Forecast skill</div>
  <h2>Global reach lands inside the noise</h2>
  <div class="prose">
  <p>At 24 hours the two local seeds score <strong>174.7</strong> and <strong>173.3</strong>.
  The globally-connected model scores <strong>173.4</strong> &mdash; between them. Whatever
  160-hop reach buys, it is smaller than the difference between running the same model twice.</p>
  <p>The control GNN is in another regime entirely: <strong>+67% at 24 h</strong>, and worse
  than persistence at 6 h and beyond 72 h. Its measured receptive field is 9 hops, less than
  the local model's 20, so it was never a test of non-locality.</p>
  </div>
  <figure>
    {rmse}
    <figcaption>z500 area-weighted RMSE on held-out test 2020, log lead axis, lower is better.
    A and B share a colour because they are the same model at different seeds.</figcaption>
  </figure>
  <div class="tbl"><table>
    <thead><tr><th class="num">lead</th><th class="num">A · local s0</th>
    <th class="num">B · local s1</th><th class="num">seed spread</th>
    <th class="num">D · d=8</th><th class="num">D vs A/B</th>
    <th class="num">C · GNN</th></tr></thead>
    <tbody>{ladder}</tbody>
  </table></div>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">Methodology</div>
  <h2>The noise floor is not one number</h2>
  <div class="prose">
  <p>The selection metric put seed-to-seed spread at <strong>0.21%</strong>, and that figure has
  been carrying a lot of weight in this milestone's planning. On the evaluation metric it holds
  only at short leads. It grows with lead time, reaching <strong>12.2% at 15 days</strong>.</p>
  <p>This changes what is claimable. At 24 h a 1% effect is near the edge of resolution with two
  seeds. At 240 h and beyond, <em>nothing</em> under ~10% is resolvable at n=2 &mdash; which
  means D's apparent 5&ndash;6% long-lead deficit against A is not a finding, it is inside the
  noise.</p>
  </div>
  <figure>
    {spread}
    <figcaption>|A − B| as a percentage of their mean, by lead time. Same model, same data,
    same recipe; only the random seed differs.</figcaption>
  </figure>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">Error growth</div>
  <h2>Doubling times, and one arm outside the band</h2>
  <div class="prose">
  <p>Perturbation growth is measured directly: perturb the initial state and track how the two
  trajectories separate per forecast window. The real atmosphere doubles synoptic errors in
  roughly <strong>1.5&ndash;2.5 days</strong>.</p>
  <p>The three local-rule arms sit in or near that band ({dA}, {dB}, {dD} days). The control GNN
  doubles in <strong>{dC} days</strong> &mdash; roughly twice as fast as the atmosphere, which
  is the signature of a model whose own states drift off-distribution, and matches its collapse
  beyond 72 h.</p>
  </div>
  <figure>
    {growth}
    <figcaption>Error-doubling time in days, from sustained per-window perturbation growth.
    Longer is slower growth &mdash; which is not automatically better: a model far above the
    1.5&ndash;2.5 day band is over-damped.</figcaption>
  </figure>
  <div class="note">
    <div class="eyebrow">Seeds move this too</div>
    <p>A and B differ by <strong>12%</strong> on doubling time ({dA} vs {dB} days) despite being
    the same model. Growth-rate comparisons across architectures need the same error bars the
    RMSE comparisons do.</p>
  </div>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">A metric artefact, confirmed</div>
  <h2>Two-metre temperature was being judged unfairly</h2>
  <div class="prose">
  <p>Phase 2c reported 2 m temperature at <strong>&minus;31% skill</strong> at 24 h &mdash; the
  only channel worse than persistence, and an apparent failure of the solar forcing. Scoring it
  at every 6-hour lead instead of only at multiples of 24 shows what was happening.</p>
  <p>Persistence at a 24-hour multiple compares the <em>same local solar time</em>, so it
  reproduces the diurnal cycle for free. For a diurnally-dominated surface field that makes it
  an unusually strong baseline at exactly those leads &mdash; and only those. Mean skill
  <strong>&minus;43.5% on the 24 h grid, +16.8% off it</strong>: a 60-point sawtooth. z500 shows
  no such pattern, so this is specific to the field, not an evaluation bug.</p>
  </div>
  <figure>
    {t2m}
    <figcaption>Skill vs persistence at every 6 h lead, arm A. The 2 m temperature sawtooth
    dips at 24, 48 and 72 h; z500 declines smoothly through the same leads.</figcaption>
  </figure>
</section>

<section>
  <hr class="rule">
  <div class="eyebrow">What this does not settle</div>
  <h2>Open, and honestly so</h2>
  <div class="cols prose">
    <div>
      <h3>The placebo arm is still training</h3>
      <p>E &mdash; parameter-identical to D, with no extra reach &mdash; is the arm that
      separates reach from capacity. Until it lands, D vs A conflates the two.</p>
      <h3>A diagnostic that did not work</h3>
      <p>The val-split probe was meant to separate rollout instability from over-regularisation.
      It compares a one-window prediction against a two-window target, so its two halves are not
      the same task and its numbers are excluded here. Found while writing this page; the fix is
      a one-line change to the loader.</p>
    </div>
    <div>
      <h3>The long-range channel is isotropic</h3>
      <p>The ring computes mean minus centre &mdash; scalar diffusion. The local stencil also
      carries direction. So this tests isotropic long-range information; a directional ring
      gradient is the follow-up.</p>
      <h3>One radius, not a curve</h3>
      <p>d = 8 is the far extreme. It answers "does global reach help?" and not "at what radius
      does reach start to matter?" &mdash; which would need d = 2 and d = 4.</p>
    </div>
  </div>
</section>

<footer class="mono">
  weather-nca · generated by scripts/report_2d.py from the phase-2d evaluation logs
</footer>
</div>
"""


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    for h, a, b, d in zip(LEADS, A, B, D):
        print(f"  {h:>4}h  seed spread {100*abs(a-b)/((a+b)/2):>5.1f}%   "
              f"D vs A/B {100*(d/((a+b)/2)-1):>+6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
