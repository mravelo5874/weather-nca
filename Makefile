# Use the local venv if it exists, otherwise whatever python is on PATH.
PY := $(shell test -x .venv/Scripts/python.exe && echo .venv/Scripts/python.exe \
        || (test -x .venv/bin/python && echo .venv/bin/python) || echo python)

.PHONY: test smoke train eval cache mesh benchmark install

install:
	$(PY) -m pip install -e ".[dev]"

# Unit tests. Must stay under 60s -- every phase runs this before it runs on cloud.
test:
	$(PY) -m pytest tests/ -q

# Full train -> eval cycle on n_sub=3 (642 nodes) with a month of data. Target under 2 min.
# Uses synthetic data so it needs no network.
smoke:
	$(PY) -m wnca.cli train --smoke --set data.source='"synthetic"'
	$(PY) -m wnca.cli eval  --smoke --set data.source='"synthetic"'

# Real-data smoke, for when the ERA5 path itself is what you want to exercise.
smoke-era5:
	$(PY) -m wnca.cli train --smoke
	$(PY) -m wnca.cli eval  --smoke

mesh:
	$(PY) -m wnca.cli mesh --config $(CONFIG)

cache:
	$(PY) -m wnca.cli cache --config $(CONFIG)

benchmark:
	$(PY) -m wnca.cli benchmark --config $(CONFIG)

train:
	$(PY) -m wnca.cli train --config $(CONFIG)

eval:
	$(PY) -m wnca.cli eval --config $(CONFIG)
