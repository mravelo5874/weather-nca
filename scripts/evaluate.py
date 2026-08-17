#!/usr/bin/env python
"""Evaluate entry point: python scripts/evaluate.py --config configs/<phase>.yaml [--smoke]."""
import sys
from wnca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["eval", *sys.argv[1:]]))
