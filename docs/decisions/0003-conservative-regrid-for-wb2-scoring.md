# 0003 — Conservative regrid for WeatherBench-2 scoring

**Date:** 2026-08-16
**Status:** accepted, implemented in `mesh/regrid.py`

## Decision

Scoring against WB2 happens **on the 1.5° lat-lon grid**, not on the mesh. The mesh → lat-lon
operator area-averages the piecewise-linear (P1) mesh interpolant over each target cell, using
a 3×3 quadrature per cell with cos(lat) weights. Rows sum to 1.

## Why

WB2 probabilistic numbers are published at 1.5°. Scoring on the mesh and comparing to a 1.5°
leaderboard number is not a comparison — it is two different metrics with the same name. A
nearest-neighbour regrid would bias the RMSE; a row-stochastic area-weighted one does not, and
reproduces constants exactly.

Conveniently, the WB2 ingest store (`240x121`) **is** the 1.5° grid, so ingest and scoring grids
coincide.

## The comparability caveat, recorded so nobody rediscovers it

WB2 scores operational models against **IFS analysis** and ML models against **ERA5**. This
project is on the ERA5 side. Therefore:

- the **GenCast** comparison is like-for-like;
- the **IFS ENS** comparison is slightly *favourable* to us, because ERA5 is our training target
  and is not what IFS ENS was scored against.

Do not quote the IFS ENS row as a clean win. This is repeated in a comment in `eval/wb2.py`.
