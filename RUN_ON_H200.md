# H200 Run Runbook — Salient ImageNet Recall Comparison

Concrete checklist for running the SI comparison on a rented H200 / H100 / A100 80 GB and shipping results back. For background and rationale, see `SALIENT_IMAGENET_EXPERIMENT.md`.

Total wall time: **~10–12 h on H200 SXM 141 GB** (~$48 at $4/h). Total human time: ~30 minutes setup + intermittent monitoring.

---

## 1. Rent the right GPU

Pick **H200 SXM 141 GB** (best). Acceptable fallback: H100 SXM5 80 GB.

**Reject** anything with < 80 GB VRAM — forces 8-bit FLUX + CPU offload and the run takes days instead of hours.

Providers (any on-demand H200 is fine):
- Lambda Labs — `lambdalabs.com/service/gpu-cloud`
- RunPod — `runpod.io`
- CoreWeave

Confirm before paying:
- [ ] 80+ GB VRAM (`nvidia-smi` shows ≥ 80000 MiB)
- [ ] Single-tenant
- [ ] ≥ 200 GB local SSD
- [ ] ≥ 200 Mbps internet (~50 GB of model weights to download)

---

## 2. Set up the machine (~5 min)

SSH in, then:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone the repo (replace with your fork/branch as needed)
git clone https://github.com/<your-account>/coralation-analisis.git
cd coralation-analisis
git checkout feat/probe-pipeline

# HuggingFace token (must have access to FLUX.2-klein and Qwen-VL gated repos)
echo "hf_YOUR_TOKEN" > .token

# Python deps
uv sync
```

Accept gated terms in a browser **once** for these repos (the page will say "You have been granted access" after):
- https://huggingface.co/datasets/ILSVRC/imagenet-1k
- https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv
- https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

Sanity check the GPU:

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
# Expect: NVIDIA H200, 143771 MiB  (or H100 80GB / A100 80GB)
```

---

## 3. Launch the run (~10–12 h)

```bash
bash scripts/run_si_h200.sh
```

The script is a guarded wrapper that:
- Refuses to start if `.token` is missing or GPU has < 80 GB VRAM
- Runs `main.py` with the canonical recall-comparison flags
- Backgrounds the process via `nohup`, writes PID to `run.pid`
- Sends stdout/stderr to `stdout.log` and DEBUG logs to `salient_imagenet_run.log`

Monitor from a second SSH session:

```bash
# Live log tail
tail -f salient_imagenet_run.log

# How many classes done?
ls output/salient_imagenet_run/checkpoints/resnet50/ 2>/dev/null | wc -l
# Expect this to grow by ~30 every ~50 min (one batch)

# GPU utilisation (should be 80-100% during FLUX phase)
watch -n 5 nvidia-smi
```

If something dies, **just re-run `bash scripts/run_si_h200.sh`** — already-completed classes load from disk and the worst-case loss is one batch (~30 classes, ~50 min).

---

## 4. Collect results (~2 min after main run finishes)

When `output/salient_imagenet_run/reports/feature_catalog.json` exists, the main run is done. Then:

```bash
bash scripts/collect_si_results.sh
```

This:
1. Runs `compare_salient_imagenet.py` (CPU, ~1 min) → writes `salient_imagenet_comparison.{json,md}`
2. Tars the deliverables into `si_results_YYYYMMDD.tgz` (~10–80 MB)

---

## 5. Ship the tarball back

Pick whichever you prefer (in roughly increasing privacy):

```bash
# A) Direct scp from your laptop
scp $USER@$HOST:~/coralation-analisis/si_results_*.tgz ~/Workspace/

# B) Public 24 h link (no account needed)
curl --upload-file si_results_*.tgz https://transfer.sh/si_results.tgz

# C) Cloud bucket
rclone copy si_results_*.tgz gdrive:si_run/
aws s3 cp si_results_*.tgz s3://your-bucket/
gh release create si-run-$(date +%Y%m%d) si_results_*.tgz --notes "SI run"
```

---

## 6. Stop billing

After the tarball is safely on your laptop:

```bash
# On the rented machine: nothing to do, just terminate it from the provider's web UI
```

**Do this immediately** — H200 is ~$4/h whether or not anything is running.

---

## 7. Back on your laptop

```bash
cd ~/Workspace/coralation-analisis
mkdir -p output/salient_imagenet_run
tar -xzf ~/Workspace/si_results_*.tgz -C ./

# The headline artifact for the paper:
cat output/salient_imagenet_run/reports/salient_imagenet_comparison.md

# Threshold sensitivity sweep (still on CPU, ~1 min each)
for t in 0.40 0.45 0.50 0.55; do
    uv run python scripts/compare_salient_imagenet.py \
        --classifier resnet50 \
        --catalog output/salient_imagenet_run/reports/feature_catalog.json \
        --out-json output/salient_imagenet_run/reports/cmp_t${t}.json \
        --out-md   output/salient_imagenet_run/reports/cmp_t${t}.md \
        --threshold $t
done
```

You now have per-threshold recall + precision tables to drop into the paper.

---

## Failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `bash scripts/run_si_h200.sh` exits "GPU has < 80 GB" | Wrong instance type | Resize to H100 80 GB / H200, re-run script |
| `bash scripts/run_si_h200.sh` exits "no .token" | Token not written | `echo "hf_YOUR_TOKEN" > .token` and re-run |
| `403` from HuggingFace mid-run | Gated terms not accepted | Open the gated-model URL in browser, click "Agree", re-run script (resume kicks in) |
| GPU util stuck < 30 % during FLUX phase | 8-bit offload accidentally on | Check the launched command includes `--no-8bit-editor`; the script enforces it |
| `nvidia-smi` shows 0 % util but process alive | Stuck downloading model weights | First-class run takes ~10 min for FLUX/VLM weights to download. Be patient. |
| Disk fills up | Edited image cache is large | `rm -rf output/salient_imagenet_run/images/` is safe between runs (checkpoints live in `checkpoints/`) |

---

## What you actually need to bring home

The tarball will contain:

```
output/salient_imagenet_run/reports/
    analysis_results.json              (Algorithm 1 raw output)
    feature_catalog.json               (per-class real/bias_spurious buckets)
    salient_imagenet_comparison.json   (per-class recall/precision)
    salient_imagenet_comparison.md     (human-readable headline table)
output/salient_imagenet_run/checkpoints/  (per-class JSONs — useful for re-running comparison with different thresholds without re-running pipeline)
salient_imagenet_run.log               (full DEBUG log)
```

The `images/` and `dataset_hf/` directories are **not** in the tarball — they can be many GB and are not needed for the paper.
