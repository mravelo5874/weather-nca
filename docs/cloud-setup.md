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

## Choosing a machine — measured on an L4, 2026-08-18

Everything below is measured, not extrapolated. The earlier "15–25× faster than the local
1660 Ti" guess was **wrong by about 5×**.

**L4 (g2-standard-8) against the local GTX 1660 Ti**, 28 channels, `hidden_dim=512`, B=8,
20 sub-steps, one forecast step forward+backward:

| | ms/step | speedup |
|---|---|---|
| 1660 Ti, fp32 | 3,722 | 1.0× |
| L4, fp32 | 1,294 | **2.9×** |
| L4, AMP | 837 | **4.4×** |

**AMP pays, finally.** 1.50–1.55× at `hidden_dim=512`, 1.36–1.42× at 256. This is the first
speedup AMP has ever produced anywhere in the project — it is inert on Turing, and it only
works at all because of the sparse-operator fp32 fix (`torch.sparse.mm` has no half kernel).

**`torch.compile` is a dead end.** With triton present and inductor working: **0.98×**. The
workload is compute-bound, not launch-bound, exactly as the local `cudagraphs` result
suggested. Do not spend more time on it.

**Larger batches buy nothing.** Throughput is flat in batch size:

| hidden | B | AMP ms/step | peak GB | samples/s |
|---|---|---|---|---|
| 512 | 8 | 837 | 1.29 | 9.6 |
| 512 | 16 | 1,784 | 2.56 | 9.0 |
| 512 | 32 | 3,605 | 5.09 | 8.9 |

Cost is linear in `B`, as measured locally. So **memory headroom is not a lever** — 23 GB is
heavily over-provisioned for a 5 GB peak, and a smaller/cheaper card loses nothing. Keep
`batch_size: 8` unless the optimizer wants otherwise.

**`hidden_dim` 512 vs 256 costs 1.56×**, not the 4× that `hidden_dim²` scaling implies. The MLP
is not the sole bottleneck at this size, which makes 512 more affordable than the local
profile suggested.

**GCS read: 35 MB/s → 65 GB in ~0.5 h.** Measured from `us-east1`, *not* co-located with the
bucket, and still fast. Region-matching matters less than assumed — capacity availability is
the better reason to pick a zone.

### Capacity is the real constraint

`nvidia-l4` was **stocked out in all three `us-central1` zones**. Any launch script must try
several zones:

```bash
for z in us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d us-west1-a; do
  gcloud compute instances create wnca-train --zone="$z" ... && break
done
```

### Image family

`pytorch-latest-gpu` no longer exists. Current families:

```bash
gcloud compute images list --project=deeplearning-platform-release --format="value(family)" | sort -u
# pytorch-2-9-cu129-ubuntu-2204-nvidia-580   <- used here
```

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

**Do not inline shell commands into `gcloud compute ssh --command` from PowerShell.** Nested
quoting is mangled by both PowerShell and cmd before gcloud sees it, and the failures are
confusing (`unrecognized arguments`, half-parsed Python). Put the work in a script, commit it,
and invoke the file. `scripts/cloud_bench.sh` exists for exactly this reason.

Also: `git pull` on the instance will abort if anything is modified (e.g. `chmod +x`), and the
run silently continues with the OLD script. Use `git reset --hard && git pull`.

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

### The two layers of restart

**Inside the instance: `scripts/spot_train.sh`.** Resumes from any existing checkpoint, mirrors
results to GCS after every attempt, and on a genuine crash retries at a **lower learning rate** —
retrying an identical diverged configuration just diverges again, which is exactly what
happened to phase 2b′. It exits 75 (`EX_TEMPFAIL`) on preemption so the outer layer can tell
"the VM went away" from "the run failed".

```bash
./scripts/spot_train.sh configs/phase2c_full.yaml gs://YOUR_BUCKET/wnca
```

**Outside: a watchdog**, because a script on the instance cannot recreate the instance it is
dying with. Simplest version, run locally:

```bash
while true; do
  state=$(gcloud compute instances describe wnca-train --zone=us-central1-a             --format="value(status)" 2>/dev/null || echo GONE)
  if [[ "$state" != "RUNNING" ]]; then
    echo "instance is $state -- recreating"
    gcloud compute instances create wnca-train ... --provisioning-model=SPOT
    gcloud compute ssh wnca-train --command       "cd weather-nca && nohup ./scripts/spot_train.sh configs/phase2c_full.yaml gs://YOUR_BUCKET/wnca > train.log 2>&1 &"
  fi
  sleep 120
done
```

A managed instance group with autohealing is the production-grade version of the same idea and
is worth it if runs stretch to days.

**The state that must survive is already handled** — checkpoint (weights, optimizer, epoch,
step, LR position, history), the cache, and `metrics.jsonl` — provided `GCS_PREFIX` is set so
the mirror happens off-instance. Without it, a reclaimed VM takes the run's only copy with it.

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
- [ ] Zone fallback list in the launch command — `nvidia-l4` was stocked out in all three
      `us-central1` zones
- [ ] `make test` and `make smoke` pass **on the instance**
- [ ] `wnca benchmark` run, and the epoch budget chosen from it
- [ ] `train.amp: true` — free 2–3× on Ampere and later, inert locally. **Tested**: sparse
      operators are forced to fp32 internally because `torch.sparse.mm` has no half kernel;
      without that fix AMP crashed on the first step
- [ ] `batch_size` left at 8 — measured: throughput is flat in batch size, so memory headroom
      is not a lever
- [ ] Auto-shutdown backstop set
- [ ] `scripts/spot_train.sh` used rather than a bare `wnca train`, with a GCS prefix set
- [ ] Outer watchdog running, or a managed instance group configured
- [ ] Quota requested via CLI (more reliable than console navigation):
      `gcloud quotas preferences create --service=compute.googleapis.com --quota-id=GPUS-ALL-REGIONS-per-project --preferred-value=1 --project=PROJECT --email=YOU`

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
