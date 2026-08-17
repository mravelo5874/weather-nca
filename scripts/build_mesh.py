#!/usr/bin/env python
"""Build and cache the icosahedral mesh and its perception coefficients."""
import sys
from wnca.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["mesh", *sys.argv[1:]]))
