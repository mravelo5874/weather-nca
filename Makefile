.PHONY: test smoke train eval

# Unit tests. Must run in under 60s on the 1660 Ti (see docs/milestone-2-plan.md).
test:
	pytest tests/ -v

# Full train -> eval cycle on n_sub=3 (642 nodes), one month of data. Target < 2 min.
# Run before every phase goes to cloud.
smoke:
	python scripts/train.py --config configs/base.yaml --smoke
	python scripts/evaluate.py --config configs/base.yaml --smoke

train:
	python scripts/train.py --config $(CONFIG)

eval:
	python scripts/evaluate.py --config $(CONFIG)
