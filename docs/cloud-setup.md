# Running M2 phases on Google Cloud

Written for someone who has not used cloud compute before. Read the **Cost control** section
before anything else — it is the only part where a mistake costs real money.

---

## Why GCP for this project specifically

Not brand preference. WeatherBench-2 lives at `gs://weatherbench2/`, and the phase-2c cache is
65 GB of data pulled from it.

| | local machine | GCE instance in the bucket's region |
|---|---|---|
| build the 65 GB cache | ~15 h streaming | minutes to ~1 h |
| egress cost | — | none for same-region reads |

If the variable set, level list, or `cache_dtype` ever changes, the cache is rebuilt. Locally
that is a day each time. **That difference is the whole argument.** If we were certain never to
rebuild, the cheaper per-hour renters (Vast.ai, RunPod) would win instead — see the plan's
compute notes.

---

## Division of labour

| step | who |
|---|---|
| create account, enter billing details | **you** (browser) |
| set the budget cap | **you** (browser) |
| request GPU quota | **you** (browser form) |
| `gcloud auth login` | **you** (opens a browser) |
| everything after that | the agent can drive it via the CLI |

Once authentication exists on this machine, creating instances, syncing the repo, launching and
monitoring runs, pulling results back and deleting the instance are all ordinary shell commands.

---

## Cost control — read this first

An idle GPU costs exactly as much as a busy one, and a forgotten instance bills 24/7. Almost
every cloud horror story is a forgotten instance, not an expensive run.

**Do these before launching anything:**

1. **Billing → Budgets & alerts → Create budget.** Set a monthly cap you are comfortable
   losing, with alerts at 50/90/100%. Alerts notify; they do not stop spending. Treat the
   number as a smoke detector, not a circuit breaker.
2. **Prefer Spot instances.** 60–90% cheaper. They can be reclaimed with ~30 s notice, which
   this repo already handles (see below).
3. **`delete`, do not `stop`.** A stopped instance still bills for its disk.
4. **Set an auto-shutdown as a backstop**, so a crashed run cannot idle for days:
   ```bash
   # on the instance: shut down after 12 h no matter what
   sudo shutdown -h +720
   ```
5. **Check for strays** before you finish for the day:
   ```bash
   gcloud compute instances list        # anything RUNNING is costing money
   ```

---

## One-time setup

### 1. Account and project

Create a Google Cloud account (the free trial normally includes credits), then a project.
Note the project ID — it is not the same as the display name.

### 2. Install the CLI

Download the Google Cloud CLI installer for Windows, then:

```bash
gcloud init                       # pick account + project
gcloud auth login                 # opens a browser -- must be done by a human
gcloud config set project YOUR_PROJECT_ID
gcloud config set compute/zone us-central1-a
```

### 3. Enable the APIs

```bash
gcloud services enable compute.googleapis.com storage.googleapis.com
```

### 4. Request GPU quota — do this early

**A new account has a GPU quota of zero.** You cannot launch a GPU instance until quota is
granted, and approval takes anywhere from an hour to a day.

IAM & Admin → Quotas → filter for `GPUS_ALL_REGIONS` → request an increase (1–4 is plenty).
Request Spot/preemptible quota too if it is listed separately.

### 5. The bucket's region -- measure it, do not look it up

The WeatherBench-2 bucket grants **object** reads but not **bucket metadata** reads, so

```bash
gsutil ls -L -b gs://weatherbench2      # AccessDeniedException: 403 storage.buckets.get
```

fails even when fully authenticated. That is expected and harmless: `storage.objects.list` and
`storage.objects.get` both work, which is all the pipeline needs.

So the location cannot be looked up. Instead:

1. **Start in `us-central1`.** Google Research public datasets are conventionally US
   multi-region, and us-central1 also has the best GPU and spot availability.
2. **Verify with a throughput test on the instance** -- a real measurement takes a minute:

```bash
time gsutil -m cp -r   gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr/geopotential/0.0.0.0   /tmp/
```

Hundreds of MB/s means in-region and the cache build will be fast. Home-connection speeds mean
cross-region -- try `us-east1` or `us-west1` before committing to a 65 GB build.

---

## Choosing a machine

Driven by what was measured, not by what looks impressive:

- **Peak GPU memory is 2.49 GB** across every M2 configuration tried. An 80 GB card is wasted
  money. We are buying throughput, not capacity.
- **~80% of a sub-step is the update MLP**, scaling with `hidden_dim²`. Large dense matmuls,
  which is what tensor cores are for.
- **Turing (the local 1660 Ti) has no tensor cores**, so `train.amp` is inert locally. On
  Ampere or newer it is a straight 2–3×, and it is already wired up — just set `train.amp: true`.
- Memory being free means **a much larger `batch_size`** is available on a cloud card, which
  also improves the matmul shapes. Worth sweeping once on the instance.

A single A100 40 GB, or a smaller Ampere card, is the sensible starting point. **Benchmark
before committing** — the "15–25× faster" figure in the findings is an extrapolation, and this
project has twice punished the habit of trusting a proxy over a direct measurement.

---

## Per-run workflow

```bash
# 1. create a spot instance with a GPU and enough disk for the cache
gcloud compute instances create wnca-train \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=250GB --boot-disk-type=pd-ssd \
  --provisioning-model=SPOT --metadata=install-nvidia-driver=True

# 2. get the code on it
gcloud compute ssh wnca-train --command "git clone <repo-url> weather-nca"
gcloud compute ssh wnca-train --command "cd weather-nca && pip install -e '.[dev]'"

# 3. verify the environment BEFORE spending GPU time on a real run
gcloud compute ssh wnca-train --command "cd weather-nca && make test && make smoke"

# 4. measure, then decide the epoch budget from the measurement
gcloud compute ssh wnca-train --command \
  "cd weather-nca && wnca benchmark -c configs/phase2c_full.yaml"

# 5. build the cache (resumable; safe to interrupt)
gcloud compute ssh wnca-train --command \
  "cd weather-nca && nohup wnca cache -c configs/phase2c_full.yaml > cache.log 2>&1 &"

# 6. train, detached so an SSH drop does not kill it
gcloud compute ssh wnca-train --command \
  "cd weather-nca && nohup wnca train -c configs/phase2c_full.yaml > train.log 2>&1 &"

# 7. watch
gcloud compute ssh wnca-train --command "tail -f weather-nca/train.log"

# 8. pull results back
gcloud compute scp --recurse wnca-train:~/weather-nca/runs ./runs

# 9. DELETE (not stop)
gcloud compute instances delete wnca-train --quiet
```

Steps 3 and 4 are not optional ceremony. `make smoke` catches a broken environment in two
minutes instead of two hours, and the benchmark is what stops us guessing an epoch budget —
the mistake that made phase 2a's epoch count arbitrary.

---

## Spot preemption

A spot instance can be reclaimed with roughly 30 seconds of notice. The repo already handles
the data side of that:

- **SIGTERM handler** — the training loop catches it, checkpoints to `preempted.pt` at the next
  safe point, and exits cleanly.
- **`wnca train --resume auto`** — restores weights, optimizer state, epoch, step and the
  LR-schedule position. Losing `step` would silently restart the cosine decay at full learning
  rate; it is restored deliberately. Tested, and used for real when phase 2b′ diverged.
- **Chunk-resumable cache**, including the in-place normalization pass. Interrupting that pass
  used to double-normalize the finished splits — silent corruption, `std` 1.00 → 2.80 with no
  error. Fixed and regression-tested.
- **`metrics.jsonl`** is written every epoch and survives without wandb.

**The remaining gap:** nothing restarts the instance automatically. If a spot instance is
reclaimed mid-run it stays dead until someone notices. Before any long unattended run, add a
wrapper that recreates the instance and relaunches with `--resume auto`. Worth writing once.

---

## Data: two options for the cache

**Rebuild on the instance** (simple). ~1 h in-region against ~15 h locally. Fine when the
instance is long-lived.

**Persist it** (better across instances). Build once, then keep it:

```bash
# to a GCS bucket of your own
gsutil -m rsync -r ~/weather-nca/wnca_cache gs://YOUR_BUCKET/wnca_cache
# or on a persistent disk that survives instance deletion
gcloud compute disks create wnca-cache --size=250GB --type=pd-ssd
```

A persistent disk is the better fit for spot instances: the disk outlives the VM, so a
reclaimed instance does not cost a rebuild. Note `data.cache_dtype: float16` halves the cache
from 65 GB to 33 GB, which matters when paying for storage.

---

## Checklist before the first real run

- [ ] Budget cap set, with alerts
- [ ] GPU quota granted (requested days earlier, not the same morning)
- [ ] Bucket region identified; instance launched in a matching zone
- [ ] `make test` and `make smoke` pass **on the instance**
- [ ] `wnca benchmark` run, and the epoch budget chosen from it
- [ ] `train.amp: true` — free 2–3× on Ampere and later, inert locally
- [ ] `batch_size` raised to use the available memory, and re-benchmarked
- [ ] Auto-shutdown backstop set
- [ ] A plan for what happens if the spot instance is reclaimed overnight

---

## What was measured locally, and what to re-measure

Carry none of these across as assumptions:

| quantity | local (GTX 1660 Ti) | re-measure? |
|---|---|---|
| step, 28 ch, `hidden_dim=512`, 20 sub-steps | 3,722 ms | yes |
| same at 40 sub-steps | 7,458 ms (exactly 2.0×) | yes |
| peak GPU memory | 2.49 GB | yes, then raise `batch_size` |
| MLP / perception split | ~80% / ~20% | probably holds |
| `torch.compile` | inductor **unavailable on Windows** (no Triton); `cudagraphs` gave ~1.03× | **yes — inductor works on Linux and is untested** |

That last row is the one most likely to change on the cloud instance, and it is free to test.
