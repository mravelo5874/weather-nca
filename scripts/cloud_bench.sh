#!/usr/bin/env bash
# First-run validation and benchmark for a fresh cloud instance.
#
# Answers, in order of how much they change the plan:
#   1. does the code run on Linux at all (it has only ever run on Windows)
#   2. how fast is a step here, so every local extrapolation can be replaced
#   3. does AMP actually pay (it is inert on Turing and has never produced a speedup)
#   4. does torch.compile/inductor pay (impossible to test on Windows -- no Triton)
#   5. how fast is the WeatherBench-2 read, which sets the 65 GB cache build time
set -uo pipefail
REPO="${REPO:-https://github.com/mravelo5874/weather-nca.git}"
cd ~ || exit 1

echo "=============== 1. ENVIRONMENT ==============="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c 'import torch,platform;print("torch",torch.__version__,"| cuda",torch.cuda.is_available(),"|",platform.platform())'
python3 -c 'import triton;print("triton",triton.__version__)' 2>/dev/null || echo "triton: MISSING (inductor will not work)"

echo; echo "=============== 2. REPO + DEPS ==============="
[ -d weather-nca ] || git clone -q "$REPO" weather-nca
cd weather-nca || exit 1
# The deep-learning images ship a setuptools too old for PEP 660, so an editable install fails
# with "build backend is missing the 'build_editable' hook". Upgrade first, and fall back to a
# regular install rather than silently continuing with no package.
pip install -q --upgrade pip setuptools wheel 2>&1 | tail -1
pip install -q -e ".[dev]" 2>&1 | tail -2 || pip install -q ".[dev]" 2>&1 | tail -2
python3 -c "import wnca, pytest, gcsfs; print('wnca importable')" || { echo "INSTALL FAILED"; exit 1; }

echo; echo "=============== 3. UNIT TESTS (Linux compat) ==============="
python3 -m pytest tests/ -q 2>&1 | tail -3

echo; echo "=============== 4. GCS READ THROUGHPUT ==============="
python3 - <<'PY'
import time, gcsfs
fs = gcsfs.GCSFileSystem(token="anon")
base = "weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr/geopotential"
files = [f for f in fs.ls(base) if not f.endswith(".json")][:6]
t0, n = time.time(), 0
for f in files:
    n += len(fs.cat_file(f))
dt = time.time() - t0
print(f"  read {n/1e6:.1f} MB in {dt:.1f}s -> {n/1e6/dt:.0f} MB/s")
print(f"  implies 65 GB cache stream in ~{65000/(n/1e6/dt)/3600:.1f} h of pure transfer")
PY

echo; echo "=============== 5. BENCHMARK SWEEP ==============="
python3 - <<'PY'
import time, itertools, numpy as np, torch
from wnca.config import load_config
from wnca.train.phases import setup
from wnca.data.forcing import SolarForcing

base = {"data":{"source":"synthetic","max_steps_per_split":40,
                "train_years":[2015],"val_years":[2016],"test_years":[2017]}}
print(f"{'hidden':>7} {'batch':>6} {'amp':>5} | {'ms/step':>9} {'peak GB':>8} {'samples/s':>10}")
print("-"*58)
results = {}
for hidden, B, amp in itertools.product((256, 512), (8, 16, 32), (False, True)):
    ov = dict(base); ov["model"]={"hidden_dim":hidden}; ov["train"]={"batch_size":B,"amp":amp}
    cfg = load_config(None, overrides=ov)
    try:
        mesh, cache, model, _, dev = setup(cfg, verbose=False)
        s = cache.split("train").array
        cur = torch.from_numpy(np.array(s[1:1+B],dtype=np.float32)).to(dev)
        prev= torch.from_numpy(np.array(s[0:B],dtype=np.float32)).to(dev)
        st = torch.from_numpy(cache.static).float().to(dev).unsqueeze(0).expand(B,-1,-1)
        sf = SolarForcing(cache.times("train"), mesh, dev).window(torch.arange(1,1+B,device=dev),1)
        model.train(); torch.cuda.reset_peak_memory_stats()
        def once():
            with torch.autocast("cuda", enabled=amp):
                p,o = model.rollout_ensemble(model.seed(cur), st, 1, prev_phys=prev,
                                             n_members=1, return_aux=True, forcing=sf)
            (p.float().pow(2).mean()+o.float()).backward(); model.zero_grad(set_to_none=True)
        once(); torch.cuda.synchronize()
        t0=time.time()
        for _ in range(5): once()
        torch.cuda.synchronize(); dt=(time.time()-t0)/5
        pk=torch.cuda.max_memory_allocated()/1e9
        results[(hidden,B,amp)] = dt
        print(f"{hidden:>7} {B:>6} {str(amp):>5} | {dt*1000:>9.0f} {pk:>8.2f} {B/dt:>10.1f}")
    except torch.cuda.OutOfMemoryError:
        print(f"{hidden:>7} {B:>6} {str(amp):>5} | {'OOM':>9}")
    except Exception as e:
        print(f"{hidden:>7} {B:>6} {str(amp):>5} | ERROR {type(e).__name__}: {str(e)[:40]}")
    del model; torch.cuda.empty_cache()

print()
for hidden in (256,512):
    for B in (8,16,32):
        a,b = results.get((hidden,B,False)), results.get((hidden,B,True))
        if a and b: print(f"  AMP speedup  hidden={hidden} B={B}: {a/b:.2f}x")
PY

echo; echo "=============== 6. TORCH.COMPILE (inductor) ==============="
python3 - <<'PY'
import time, numpy as np, torch
from wnca.config import load_config
from wnca.train.phases import setup
cfg = load_config(None, overrides={"data":{"source":"synthetic","max_steps_per_split":40,
        "train_years":[2015],"val_years":[2016],"test_years":[2017]}})
mesh, cache, model, _, dev = setup(cfg, verbose=False)
N=len(mesh["v"]); B=8
x=torch.randn(B,N,cfg.c_state,device=dev); cond=torch.randn(B,N,cfg.c_cond,device=dev)
def t(fn,n=15):
    fn(); torch.cuda.synchronize(); t0=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/n*1000
with torch.no_grad():
    eager = t(lambda: model.nca_step(x,cond,None))
    try:
        c = torch.compile(model.nca_step, backend="inductor")
        c(x,cond,None)
        comp = t(lambda: c(x,cond,None))
        print(f"  eager {eager:.2f} ms | inductor {comp:.2f} ms -> {eager/comp:.2f}x")
    except Exception as e:
        print(f"  inductor FAILED: {type(e).__name__}: {str(e)[:120]}")
PY

echo; echo "=============== 7. END-TO-END SMOKE ==============="
python3 -m wnca.cli train --smoke --set data.source='"synthetic"' 2>&1 | tail -3
echo; echo "=============== DONE ==============="
