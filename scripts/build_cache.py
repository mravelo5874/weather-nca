#!/usr/bin/env python
"""Build the resumable mesh-projected data cache from ERA5.

Safe to interrupt and re-run: progress is recorded per split in timesteps.
"""
import sys
from wnca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["cache", *sys.argv[1:]]))
