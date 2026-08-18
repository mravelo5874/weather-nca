"""Cache build resumption, including the dangerous part: the in-place normalization pass.

Normalization is **not idempotent**. Applying it twice to a chunk produces
`((x - mu)/sd - mu)/sd`, which is silently wrong -- no exception, no NaN, just data that
trains to nonsense. The streaming pass has always been chunk-resumable; the normalization
pass was not, so a build interrupted inside it would re-encode the splits it had already
finished. At 65 GB (phase 2c) that pass is long enough for a spot preemption to land in it.

These tests kill the build mid-normalization and assert the resumed cache is byte-identical
to one built in a single go.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wnca.config import load_config
from wnca.data.cache import build_cache, cache_tag


@pytest.fixture
def synth_cfg(tmp_path):
    """Synthetic multi-channel config -- no network, small enough to build repeatedly."""
    return load_config(
        None,
        overrides={
            "phase": "cache_resume",
            "mesh": {"n_sub": 2},
            "variables": {
                "atmospheric": ["geopotential", "specific_humidity"],
                "levels": [500, 850],
                "surface": ["2m_temperature"],
                "log_transform": ["specific_humidity"],
            },
            "data": {
                "source": "synthetic",
                "cache_dir": str(tmp_path / "cache"),
                "max_steps_per_split": 40,
                "train_years": [2015],
                "val_years": [2016],
                "test_years": [2017],
            },
        },
    )


def _read(cfg, split):
    root = cfg.cache_dir / cache_tag(cfg)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    shape = tuple(man["shapes"][split])
    return np.array(np.memmap(root / f"{split}.dat", dtype=man["dtype"], mode="r", shape=shape))


def _manifest(cfg):
    root = cfg.cache_dir / cache_tag(cfg)
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def test_clean_build_is_normalized(synth_cfg, small_mesh):
    build_cache(synth_cfg, small_mesh, verbose=False)
    man = _manifest(synth_cfg)
    assert man["normalized"] is True
    for s in ("train", "val", "test"):
        assert man["norm_progress"][s] == man["shapes"][s][0]
    x = _read(synth_cfg, "train")
    assert abs(float(x.mean())) < 0.2 and 0.7 < float(x.std()) < 1.3


def test_resume_mid_normalization_matches_a_clean_build(synth_cfg, small_mesh, monkeypatch):
    """THE test. Interrupt inside the normalization pass, resume, compare to a clean build."""
    reference = {s: None for s in ("train", "val", "test")}
    build_cache(synth_cfg, small_mesh, verbose=False)
    for s in reference:
        reference[s] = _read(synth_cfg, s)

    # Second cache, same config, but crash partway through normalizing.
    import shutil

    root = synth_cfg.cache_dir / cache_tag(synth_cfg)
    shutil.rmtree(root)

    import wnca.data.cache as cache_mod

    real_encode = cache_mod.Normalizer.encode
    calls = {"n": 0}

    def flaky(self, x):
        calls["n"] += 1
        if calls["n"] == 3:  # partway through, after at least one split has progressed
            raise KeyboardInterrupt("simulated preemption")
        return real_encode(self, x)

    monkeypatch.setattr(cache_mod.Normalizer, "encode", flaky)
    with pytest.raises(KeyboardInterrupt):
        build_cache(synth_cfg, small_mesh, verbose=False)

    man = _manifest(synth_cfg)
    assert man["normalized"] is False
    assert any(v > 0 for v in man["norm_progress"].values()), "no partial progress was recorded"

    monkeypatch.undo()
    build_cache(synth_cfg, small_mesh, verbose=False)  # resume

    for s in reference:
        got = _read(synth_cfg, s)
        assert np.array_equal(got, reference[s]), f"{s} differs from a clean build after resume"


def test_resume_does_not_refit_the_normalizer(synth_cfg, small_mesh, monkeypatch):
    """Refitting on a partially normalized train split would produce meaningless statistics
    and apply them to the remaining chunks, leaving the cache internally inconsistent."""
    import wnca.data.cache as cache_mod

    real_encode = cache_mod.Normalizer.encode
    calls = {"n": 0}

    def flaky(self, x):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("simulated preemption")
        return real_encode(self, x)

    monkeypatch.setattr(cache_mod.Normalizer, "encode", flaky)
    with pytest.raises(KeyboardInterrupt):
        build_cache(synth_cfg, small_mesh, verbose=False)
    monkeypatch.undo()

    root = synth_cfg.cache_dir / cache_tag(synth_cfg)
    before = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))

    fits = {"n": 0}
    real_fit = cache_mod.fit_normalizer

    def counting_fit(*a, **k):
        fits["n"] += 1
        return real_fit(*a, **k)

    monkeypatch.setattr(cache_mod, "fit_normalizer", counting_fit)
    build_cache(synth_cfg, small_mesh, verbose=False)

    assert fits["n"] == 0, "normalizer was refitted on partially normalized data"
    after = json.loads((root / "normalizer.json").read_text(encoding="utf-8"))
    assert before["mean"] == after["mean"] and before["std"] == after["std"]


def test_completed_cache_is_not_renormalized(synth_cfg, small_mesh, monkeypatch):
    """Re-running a finished build must be a no-op, not a second pass over the data."""
    build_cache(synth_cfg, small_mesh, verbose=False)
    first = _read(synth_cfg, "train")

    import wnca.data.cache as cache_mod

    def boom(self, x):
        raise AssertionError("encode() called on an already-normalized cache")

    monkeypatch.setattr(cache_mod.Normalizer, "encode", boom)
    build_cache(synth_cfg, small_mesh, verbose=False)
    assert np.array_equal(_read(synth_cfg, "train"), first)


def test_double_normalization_would_be_detectable(synth_cfg, small_mesh):
    """Guard-rail sanity: confirm the corruption this protects against is real, so the tests
    above are not asserting something that could never happen anyway."""
    build_cache(synth_cfg, small_mesh, verbose=False)
    from wnca.data.normalize import Normalizer

    root = synth_cfg.cache_dir / cache_tag(synth_cfg)
    norm = Normalizer.load(root / "normalizer.json")
    x = _read(synth_cfg, "train")
    twice = norm.encode(x)
    assert not np.allclose(twice, x, atol=1e-3), "normalization appears idempotent -- it is not"


def test_synthetic_seed_is_stable_across_processes():
    """`hash()` is salted per process (PYTHONHASHSEED), so seeding synthetic data with it means
    a cache rebuilt in a different process holds DIFFERENT data under the SAME cache tag --
    silently, because the tag does not cover the seed.

    Checked by running the seed function in a subprocess rather than trusting the source.
    """
    import subprocess
    import sys

    from wnca.data.cache import _split_seed

    code = "from wnca.data.cache import _split_seed; print(_split_seed('train'))"
    a = int(subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout)
    b = int(subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout)
    assert a == b == _split_seed("train"), "synthetic seed differs across processes"
    assert _split_seed("train") != _split_seed("val"), "splits must not share a seed"


def test_builtin_hash_really_is_unstable():
    """The control for the test above: if `hash()` were stable, the fix would be pointless."""
    import subprocess
    import sys

    code = "print(hash('train'))"
    vals = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip() for _ in range(4)}
    assert len(vals) > 1, "hash() appears stable here -- PYTHONHASHSEED may be pinned"
