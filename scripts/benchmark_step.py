#!/usr/bin/env python
"""Benchmark one forecast step (M x n_substeps network evaluations). Run this first.

The plan's compute-cost paragraph is an estimate. This measures it; put the measured number
back into docs/milestone-2-plan.md rather than trusting the estimate.
"""
import sys
from wnca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
