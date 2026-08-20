#!/usr/bin/env python
"""Static summary figures for the phase ladder.

Produces three panels that the per-phase scorecards cannot show on their own:

  1. RMSE and skill vs lead time for each phase, so the crossover is visible rather than
     inferred from a table.
  2. Skill against persistence, with the lead where each phase stops beating it.
  3. Sustained perturbation growth per variable, as an error-doubling time, against the
     range the real atmosphere occupies.

Numbers are pasted in from the evaluation runs rather than recomputed, so this script is
cheap to re-run and never silently disagrees with the scorecards it summarises.

**All z500 numbers here are test 2018**, including phase 2c's -- 2c's own test split is 2020,
but 2b/2b'/2b-pf were scored on 2018, and a headline comparison across different years would
confound the data scale-up with a change of verification year. 2c's 2018 column is its
VALIDATION split (it drove checkpoint selection), so it is not held out; the honest reading is
that 2c's held-out 2020 numbers (24 h 174.7) and its 2018 numbers (167.6) differ by ~4%, so
selection bias is small and the ladder comparison survives it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path("media")

LEADS = np.array([6, 12, 24, 48, 72, 120, 168, 240, 360])

# z500 RMSE (m^2/s^2), test 2018, identical eval settings and start times.
PHASE0 = np.array([110.2, 169.5, 329.7, 639.1, 881.1, 1231.8, 1493.3, 1795.6, 2208.5])
PHASE2B = np.array([106.3, 158.0, 297.6, 624.0, 934.1, 1443.7, 1862.0, 2457.1, 3450.6])
# 2b' -- solar forcing + 40 sub-steps, but LR halved from epoch 3 after a divergence, so it is
# UNDERTRAINED relative to 2b (selection 0.1817 vs 0.1422). Not a clean control.
PHASE2BP = np.array([121.2, 178.8, 338.8, 684.4, 978.4, 1383.9, 1677.4, 2005.4, 2591.5])
# 2b-pushforward. The scorecard in milestone-2-findings does not carry a 240 h column.
PHASE2PF = np.array([152.4, 167.9, 282.1, 531.0, 755.6, 1085.1, 1308.7, np.nan, 1860.5])
# 2c -- 39 years, 28 channels, spectral norm + weight decay 0.1. Val split (2018); see module
# docstring for why this column and not the 2020 test split.
PHASE2C = np.array([128.4, 107.2, 167.6, 309.2, 458.8, 743.6, 949.6, 1169.8, 1475.8])
PERSIST = np.array([226.2, 361.6, 589.8, 818.2, 924.8, 1017.3, 1064.1, 1096.5, 1128.6])
CLIM = np.array([1065.2, 1064.6, 1064.6, 1064.4, 1067.7, 1073.8, 1071.6, 1063.1, 1069.9])

# Sustained per-window perturbation growth, phase 2b, by variable and level.
GROWTH = {
    "geopotential": [1.060, 1.063, 1.067, 1.073, 1.080],
    "u_wind": [1.073, 1.075, 1.076, 1.082, 1.081],
    "v_wind": [1.068, 1.076, 1.086, 1.093, 1.094],
    "temperature": [1.078, 1.080, 1.087, 1.074, 1.068],
    "humidity": [1.074, 1.077, 1.068, 1.069, 1.069],
}
GROWTH_2BP = {  # 40 sub-steps did NOT reduce this -- it rose
    "geopotential": [1.085, 1.087, 1.090, 1.085, 1.085],
    "u_wind": [1.081, 1.086, 1.094, 1.099, 1.098],
    "v_wind": [1.084, 1.088, 1.113, 1.121, 1.117],
    "temperature": [1.084, 1.087, 1.094, 1.083, 1.086],
    "humidity": [1.075, 1.091, 1.085, 1.096, 1.098],
}
GROWTH_2C = {  # sustained x1.079 overall -> 2.28 d, inside the synoptic band
    "geopotential": [1.110, 1.104, 1.112, 1.083, 1.086],
    "u_wind": [1.093, 1.100, 1.105, 1.105, 1.095],
    "v_wind": [1.093, 1.100, 1.116, 1.113, 1.103],
    "temperature": [1.068, 1.088, 1.099, 1.096, 1.111],
    "humidity": [1.102, 1.099, 1.103, 1.086, 1.078],
}
LEVELS = [850, 700, 500, 300, 250]

C0, C2B, C2BP, C2PF, C2C = "#4C6EF5", "#E8590C", "#2B8A3E", "#9C36B5", "#0B7285"


def doubling_days(per_window: float) -> float:
    """Per-6h growth -> error-doubling time in days."""
    return float(np.log(2) / np.log(per_window**4))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)

    # --- RMSE vs lead ---
    ax = axes[0]
    for y, c, lab in ((PHASE0, C0, "phase 0  (1 var, 2 yr)"),
                      (PHASE2B, C2B, "2b (28 var, 2 yr)"),
                      (PHASE2BP, C2BP, "2b' (+solar, 40 sub)*"),
                      (PHASE2PF, C2PF, "2b-pushforward"),
                      (PHASE2C, C2C, "2c (28 var, 39 yr)")):
        ax.plot(LEADS, y, "-o", ms=4, color=c, lw=2.2 if lab.startswith("2c") else 1.5, label=lab)
    ax.plot(LEADS, PERSIST, "--", color="#868E96", label="persistence")
    ax.plot(LEADS, CLIM, ":", color="#212529", label="climatology")
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("z500 area-weighted RMSE (m$^2$/s$^2$)")
    ax.set_title("Data volume dominates every other change", fontsize=11)
    ax.text(0.02, 0.02, "* 2b' undertrained: LR halved after a divergence", fontsize=7,
            color=C2BP, transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)

    # --- skill vs persistence, with each phase's crossover ---
    ax = axes[1]
    for y, c, lab in ((PHASE0, C0, "phase 0"), (PHASE2B, C2B, "2b"),
                      (PHASE2BP, C2BP, "2b'*"), (PHASE2PF, C2PF, "2b-pf"),
                      (PHASE2C, C2C, "2c")):
        ax.plot(LEADS, 100 * (1 - y / PERSIST), "-o", ms=4, color=c,
                lw=2.2 if lab == "2c" else 1.5, label=lab)
    ax.axhline(0, color="0.6", lw=1, ls="--")
    # Where does 2c stop beating persistence?
    ok = ~np.isnan(PHASE2C)
    skill2c = 100 * (1 - PHASE2C[ok] / PERSIST[ok])
    cross = float(np.interp(0.0, skill2c[::-1], LEADS[ok][::-1]))
    ax.axvline(cross, color="0.4", lw=1, ls=":")
    ax.annotate(f"2c crosses ~{cross:.0f} h\n(~{cross / 24:.1f} d)", xy=(cross, 8), fontsize=8,
                xytext=(cross - 150, 30), color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("skill vs persistence (%)")
    ax.set_title("2c holds positive skill ~3x longer than 2b", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # --- error-doubling time per variable ---
    ax = axes[2]
    ax.axhspan(1.5, 2.5, color="#2B8A3E", alpha=0.12)
    ax.text(0.02, 2.0, "real atmosphere\n(synoptic)", fontsize=7.5, color="#2B8A3E",
            va="center", transform=ax.get_yaxis_transform())
    for name, vals in GROWTH.items():
        ax.plot(LEVELS, [doubling_days(v) for v in vals], "-o", ms=3, lw=1,
                alpha=0.5, color=C2B)
    for name, vals in GROWTH_2BP.items():
        ax.plot(LEVELS, [doubling_days(v) for v in vals], "--", lw=1, alpha=0.45, color=C2BP)
    for name, vals in GROWTH_2C.items():
        ax.plot(LEVELS, [doubling_days(v) for v in vals], "-o", ms=4, lw=1.8,
                alpha=0.9, color=C2C)
    ax.plot([], [], "-o", ms=3, lw=1, color=C2B, alpha=0.6, label="2b (5 vars)")
    ax.plot([], [], "--", lw=1, color=C2BP, alpha=0.6, label="2b' (5 vars)")
    ax.plot([], [], "-o", ms=4, lw=1.8, color=C2C, label="2c (5 vars)")
    ax.axhline(doubling_days(1.033), color=C0, lw=1.2, ls="--")
    ax.text(700, doubling_days(1.033) + 0.12, "phase 0 (over-damped)", fontsize=7.5, color=C0)
    ax.invert_xaxis()
    ax.set_xlabel("pressure level (hPa)")
    ax.set_ylabel("error-doubling time (days)")
    ax.set_title("2c sits inside the atmospheric band", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    p = OUT / "ladder_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # A quick numeric echo, so the figure and the text can never drift apart.
    print(f"\n2c stops beating persistence at ~{cross:.0f} h ({cross / 24:.1f} d)")
    print(f"2c sustained growth x1.079 -> doubling {doubling_days(1.079):.2f} d "
          f"(2b-pf x1.139 -> {doubling_days(1.139):.2f} d; phase 0 x1.033 -> "
          f"{doubling_days(1.033):.2f} d)")
    print(f"\n{'lead':>6} {'2b-pf':>9} {'2c':>9} {'delta':>8}")
    for i, h in enumerate(LEADS):
        if np.isnan(PHASE2PF[i]):
            continue
        print(f"{h:>5}h {PHASE2PF[i]:>9.1f} {PHASE2C[i]:>9.1f} "
              f"{100 * (PHASE2C[i] / PHASE2PF[i] - 1):>+7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
