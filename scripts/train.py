#!/usr/bin/env python
"""Train entry point: python scripts/train.py --config configs/<phase>.yaml [--smoke]."""
import sys
from wnca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
