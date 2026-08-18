#!/usr/bin/env python
"""Static summary figures for the phase ladder.

Produces two panels that the per-phase scorecards cannot show on their own:

  1. RMSE and skill vs lead time for each phase, so the crossover is visible rather than
     inferred from a table.
  2. Sustained perturbation growth per variable, as an error-doubling time, against the
     range the real atmosphere occupies.

Numbers are pasted in from the evaluation runs rather than recomputed, so this script is
cheap to re-run and never silently disagrees with the scorecards it summarises.
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
LEVELS = [850, 700, 500, 300, 250]


def doubling_days(per_window: float) -> float:
    """Per-6h growth -> error-doubling time in days."""
    return float(np.log(2) / np.log(per_window**4))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), constrained_layout=True)

    # --- RMSE vs lead ---
    ax = axes[0]
    ax.plot(LEADS, PHASE0, "-o", ms=4, color="#4C6EF5", label="phase 0  (1 var, 2 yr)")
    ax.plot(LEADS, PHASE2B, "-o", ms=4, color="#E8590C", label="phase 2b (28 var, 2 yr)")
    ax.plot(LEADS, PHASE2BP, "-o", ms=4, color="#2B8A3E", label="phase 2b' (+solar, 40 sub)*")
    ax.plot(LEADS, PERSIST, "--", color="#868E96", label="persistence")
    ax.plot(LEADS, CLIM, ":", color="#212529", label="climatology")
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("z500 area-weighted RMSE (m$^2$/s$^2$)")
    ax.set_title("Coupling helps early, hurts late", fontsize=11)
    ax.text(0.02, 0.02, "* 2b' undertrained: LR halved after a divergence", fontsize=7,
            color="#2B8A3E", transform=ax.transAxes)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # --- skill vs persistence, with the crossover marked ---
    ax = axes[1]
    s0 = 1 - PHASE0 / PERSIST
    s2 = 1 - PHASE2B / PERSIST
    ax.plot(LEADS, 100 * s0, "-o", ms=4, color="#4C6EF5", label="phase 0")
    ax.plot(LEADS, 100 * s2, "-o", ms=4, color="#E8590C", label="phase 2b")
    ax.plot(LEADS, 100 * (1 - PHASE2BP / PERSIST), "-o", ms=4, color="#2B8A3E", label="phase 2b'*")
    ax.axhline(0, color="0.6", lw=1, ls="--")

    # Where does 2b stop beating phase 0?
    d = PHASE2B - PHASE0
    cross = np.interp(0.0, d, LEADS)
    ax.axvline(cross, color="0.4", lw=1, ls=":")
    ax.annotate(f"crossover ~{cross:.0f} h", xy=(cross, 10), fontsize=8,
                xytext=(cross + 25, 25), color="0.3",
                arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8))
    ax.set_xlabel("lead time (h)")
    ax.set_ylabel("skill vs persistence (%)")
    ax.set_title("2b wins to ~2.5 days, loses after", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # --- error-doubling time per variable ---
    ax = axes[2]
    ax.axhspan(1.5, 2.5, color="#2B8A3E", alpha=0.12)
    ax.text(0.02, 2.0, "real atmosphere\n(synoptic)", fontsize=7.5, color="#2B8A3E",
            va="center", transform=ax.get_yaxis_transform())
    for name, vals in GROWTH.items():
        ax.plot(LEVELS, [doubling_days(v) for v in vals], "-o", ms=4, label=f"2b {name}")
    for name, vals in GROWTH_2BP.items():
        ax.plot(LEVELS, [doubling_days(v) for v in vals], "--", lw=1, alpha=0.65)
    ax.plot([], [], "--", color="0.4", lw=1, label="2b' (dashed)")
    ax.axhline(doubling_days(1.033), color="#4C6EF5", lw=1.2, ls="--")
    ax.text(700, doubling_days(1.033) + 0.12, "phase 0 (z500 only)", fontsize=7.5, color="#4C6EF5")
    ax.invert_xaxis()
    ax.set_xlabel("pressure level (hPa)")
    ax.set_ylabel("error-doubling time (days)")
    ax.set_title("40 sub-steps did not slow error growth", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    p = OUT / "ladder_summary.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"wrote {p}")

    # A quick numeric echo, so the figure and the text can never drift apart.
    print(f"\ncrossover (2b stops beating phase 0): ~{cross:.0f} h")
    print(f"{'lead':>6} {'phase0':>9} {'2b':>9} {'delta':>8}")
    for i, h in enumerate(LEADS):
        print(f"{h:>5}h {PHASE0[i]:>9.1f} {PHASE2B[i]:>9.1f} {100 * (PHASE2B[i] / PHASE0[i] - 1):>+7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
