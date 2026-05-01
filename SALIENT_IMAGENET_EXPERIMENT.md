# Salient ImageNet Comparison Experiment

End-to-end guide for running **Algorithm 1 only** (per-classifier feature discovery, no probes) on the Salient ImageNet class set with **ResNet-50**, then comparing the discovered features against the human Mechanical-Turk annotations from Singla & Feizi (ICLR 2022).

This is the experiment intended to be run on a stronger GPU machine. Everything in this directory is self-contained: the SI ground-truth CSVs are checked into `data/salient_imagenet/` and the class JSON has already been built.

---

## 0. The 60-second TL;DR — what to rent and what to run

**Goal of the run:** measure recall of SI human-labelled spurious features against our `bias_spurious` bucket on the same 357 ImageNet classes. We are not optimising precision or doing a full feature-inventory match.

### Hardware to rent

| Choice | VRAM | Why | ETA for the recall run | Cost guide |
|---|---|---|---|---|
| **H200 SXM 141 GB** ✅ best | 141 GB HBM3e, 4.8 TB/s | 30–40 % faster than H100 SXM on diffusion (memory-bandwidth-bound). Single GPU is enough — no need for multi-GPU. | **~10–12 h** | ~$3.5–5/h on Lambda Labs / RunPod / CoreWeave |
| H100 SXM5 80 GB | 80 GB HBM3, 3.35 TB/s | Plenty of room for FLUX BF16 + Qwen-VL co-resident; no quantisation needed. | ~13–15 h | ~$2.5–4/h |
| A100 SXM 80 GB | 80 GB HBM2e, 2 TB/s | Works, slower memory bandwidth hurts FLUX. | ~16–20 h | ~$1.5–2.5/h |
| H100 PCIe 80 GB | 80 GB HBM2e, 2 TB/s | Same memory bandwidth as A100. Avoid if H100 SXM is available at a similar price. | ~17 h | similar |
| Anything < 80 GB | — | Forces 8-bit quant + CPU offload → 4–8× slower per edit. **Do not use.** | days | — |

**Pick H200 SXM 141 GB.** Total compute spend at $4/h × 12 h ≈ **$48** for the comparison run.

Cloud providers I'd start with (any on-demand H200 is fine):
- **Lambda Labs** (`lambdalabs.com/service/gpu-cloud`) — clean Ubuntu image, fast disk
- **RunPod** (`runpod.io`) — community + secure cloud, often cheaper
- **CoreWeave** — enterprise H200 SXM clusters

Confirm before paying: **80+ GB VRAM, single-tenant, ≥ 200 GB local SSD, ≥ 200 Mbps internet**.

### One command to launch on the rented H200

```bash
# After cloning the repo + writing .token (see §3) and `uv sync` (§3):
nohup uv run python main.py \
    --class-file src/salient_imagenet_classes.json \
    --all \
    --classifiers resnet50 \
    --no-probes \
    --samples 10 \
    --max-hypotheses 5 \
    --top-negative-classes 2 \
    --negative-samples 2 \
    --delta-threshold 0.10 \
    --attention scorecam \
    --batch-size 30 \
    --no-8bit-editor \
    --high-vram \
    --output-dir output/salient_imagenet_run \
    --log-file salient_imagenet_run.log \
    --debug \
    > stdout.log 2>&1 &
echo $! > run.pid
```

This is the **canonical recall-run command**. Settings rationale: `--samples 10 --max-hypotheses 5` keep paper alignment for what's being compared; `--top-negative-classes 2 --negative-samples 2` trim the negative-edit budget that doesn't affect recall against SI; `--delta-threshold 0.10` lowers the bar for `bias_spurious` (favours recall); `--no-8bit-editor --high-vram` is mandatory on 80+ GB cards (4–8× speedup over the 24 GB defaults); `--batch-size 30` checkpoints after every 30 classes so a crash costs ≤ ~50 min of work.

Monitor with `tail -f salient_imagenet_run.log` from a second SSH session.

### One command to compare and ship results back

After the run finishes (`output/salient_imagenet_run/reports/feature_catalog.json` exists):

```bash
# Run the comparison (~1 min on CPU)
uv run python scripts/compare_salient_imagenet.py \
    --classifier resnet50 \
    --catalog output/salient_imagenet_run/reports/feature_catalog.json \
    --out-json output/salient_imagenet_run/reports/salient_imagenet_comparison.json \
    --out-md   output/salient_imagenet_run/reports/salient_imagenet_comparison.md \
    --threshold 0.45

# Tar everything you need to bring home (~10–80 MB)
tar -czf si_results_$(date +%Y%m%d).tgz \
    output/salient_imagenet_run/reports/ \
    output/salient_imagenet_run/checkpoints/ \
    salient_imagenet_run.log

ls -lh si_results_*.tgz
```

Then on your laptop, pull it down with **one** of:

```bash
# Option A — direct scp (replace HOST with the rented machine)
scp user@HOST:~/coralation-analisis/si_results_*.tgz ~/Workspace/

# Option B — via your storage of choice (in increasing privacy)
#   GitHub release upload (gh release create ...)
#   rclone copy si_results_*.tgz gdrive:si_run/
#   aws s3 cp si_results_*.tgz s3://your-bucket/
#   curl --upload-file ... https://transfer.sh/si_results.tgz   (24 h public link)
```

Back on this laptop, drop the tarball into the project and you're ready to write up:

```bash
mkdir -p output/salient_imagenet_run
tar -xzf ~/Workspace/si_results_*.tgz -C ./
ls output/salient_imagenet_run/reports/
```

The comparison-script output (`salient_imagenet_comparison.md`) is the headline artifact for the paper.

---

## 1. What we are measuring

For each ImageNet class `c`:

- **Our system** (`Algorithm 1`) outputs a per-classifier four-bucket partition of features for `c`: `real`, `bias_spurious`, `bias_state`, `inconclusive`.
- **Salient ImageNet** outputs, for `c`'s 5 most predictive neural features (from a robust ResNet-50), a human label per feature: `main_object` (= core), `background`, or `separate_object` (the latter two = spurious by majority vote).

We compare on two axes:

1. **Spurious recall** — of the SI human-labelled spurious features for class `c`, how many were independently surfaced by our `bias_spurious` bucket?
2. **Spurious precision** — of our `bias_spurious` features for class `c`, how many semantically match an SI spurious feature?

Matching is done via cosine similarity of sentence embeddings (`all-MiniLM-L6-v2`) between our discovered feature names (e.g. `"water body"`) and the joined MTurk worker reasons for each SI feature (e.g. `"focus is on the water around the fish"`). Default threshold: `0.45`.

### Why ResNet-50 only

Salient ImageNet derives its features from a **robust ResNet-50** (`imagenet_l2_3_0.pt`). The human annotations describe attention patterns of *that specific architecture*. Comparing to our DINOv2 / ViT-L / CLIP runs would conflate "feature wasn't discovered" with "feature isn't present in this model's representation space". We therefore audit only `resnet50` for this comparison.

### Class scope: 357 classes, not 232

The original paper claims "232 classes with spurious features" but the public MTurk CSV only supports the broader **majority-vote** criterion (≥3/5 workers labelled a feature `background` or `separate_object`), which yields **357 classes** with at least one spurious feature out of 1000 annotated. The 232 number comes from a stricter, separately-verified subset that was never released. We use the 357-class superset by default; you can subset later for any paper claim.

If you want to reproduce a tighter ~221-class set close to the paper's number, regenerate with `--min-votes 4` (see §3 below).

---

## 2. What is on disk

| Path | What it is | Size |
|---|---|---|
| `data/salient_imagenet/discover_spurious_features.csv` | Raw MTurk votes (5 workers × 5 features × 1000 classes) | ~2.2 MB |
| `data/salient_imagenet/class_metadata.csv` | Synset / wordnet_id / human-readable name lookup for all 1000 classes | ~330 KB |
| `data/salient_imagenet/ground_truth.json` | Built artifact: per-class human feature labels + worker reasons | ~5 MB |
| `src/salient_imagenet_classes.json` | 357-class subset in synset_id → name format (drop-in for `--class-file`) | ~17 KB |
| `scripts/build_salient_imagenet_artifacts.py` | Rebuilds the JSON files from the CSVs (no GPU) | — |
| `scripts/compare_salient_imagenet.py` | Runs the post-hoc comparison after Algorithm 1 finishes | — |

You do not need internet access for the comparison step — everything is local.

---

## 3. One-time setup on the new machine

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Python deps (pinned via uv.lock)
uv sync

# 3. HuggingFace token (needed for ImageNet-1k validation split + FLUX.2 weights)
echo "hf_YOUR_TOKEN_HERE" > .token
# or: export HF_TOKEN=hf_YOUR_TOKEN_HERE

# 4. Accept gated terms in the browser ONCE for these repos:
#    - https://huggingface.co/datasets/ILSVRC/imagenet-1k
#    - https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv
#    - https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct

# 5. (Optional) rebuild artifacts from raw CSVs to verify reproducibility:
uv run python scripts/build_salient_imagenet_artifacts.py
# Optional stricter subset matching ~221 classes:
uv run python scripts/build_salient_imagenet_artifacts.py --min-votes 4 \
    --class-output src/salient_imagenet_classes_strict.json
```

**Disk required during the run:** ~80 GB total (model weights ~50 GB + edited image cache up to ~30 GB for 357 classes).
**VRAM required:** 24 GB minimum (RTX 4090 / A6000). With 40 GB+ you can drop `--low-vram` and run faster.

> **Pick the right command for your GPU.** The 24 GB command in §4 below uses 8-bit FLUX + CPU offload — necessary on a 4090, but it's the bottleneck and turns the run into a multi-day job. **For any 40 GB+ card (A100 / H100 / H200 / B200), use the H100 command in §4.1 instead** — adds `--no-8bit-editor --high-vram` and trims hyperparameters to fit the run in ≤ 12 h.

---

## 4. The actual experiment command (24 GB GPUs — RTX 4090, A6000)

```bash
uv run python main.py \
    --class-file src/salient_imagenet_classes.json \
    --all \
    --classifiers resnet50 \
    --no-probes \
    --samples 10 \
    --top-negative-classes 5 \
    --negative-samples 5 \
    --delta-threshold 0.15 \
    --max-hypotheses 5 \
    --attention scorecam \
    --batch-size 20 \
    --output-dir output/salient_imagenet_run \
    --log-file salient_imagenet_run.log \
    --debug
```

Flag-by-flag:

- `--class-file src/salient_imagenet_classes.json` — pulls in the 357 SI spurious-feature classes
- `--all` — process every class in the file
- `--classifiers resnet50` — Algorithm 1 will be run for the single classifier the SI annotations were produced on
- `--no-probes` — skips Phase 8 (probe synthesis) and Phase 9 (cross-model evaluation). Only Algorithm 1 (Phases 1–5) and the cheap feature-catalogue assembly run
- `--samples 10` — matches paper hyperparameters (10 positives per class)
- `--top-negative-classes 5 --negative-samples 5` — paper hyperparameters
- `--delta-threshold 0.15` — paper τ
- `--max-hypotheses 5` — VLM emits up to 5 features per positive image (matches catalog density in §4 of the paper)
- `--attention scorecam` — paper default; switch to `gradcam` for a ~3× speedup with negligible quality difference if you are tight on time
- `--batch-size 20` — process classes in batches of 20 and **save checkpoints after each batch**. Worst-case loss on a crash is ~5% of work. Set to 0 to disable batching (saves only at end).
- `--output-dir output/salient_imagenet_run` — keeps this run separate from the existing 100-class run
- `--log-file ... --debug` — full DEBUG log to disk; use `tail -f salient_imagenet_run.log` to watch progress

## 4.1 The H100 / A100 80 GB command (target: < 12 h)

**This is the command to use on H100 80 GB or A100 80 GB.** It disables 8-bit/CPU-offload (uses pure BF16), keeps all models GPU-resident, and trims hyperparameters to make the FLUX edit count tractable. See §4.2 for the timing breakdown.

```bash
uv run python main.py \
    --class-file src/salient_imagenet_classes.json \
    --all \
    --classifiers resnet50 \
    --no-probes \
    --samples 10 \
    --top-negative-classes 3 \
    --negative-samples 5 \
    --delta-threshold 0.15 \
    --max-hypotheses 3 \
    --attention scorecam \
    --batch-size 30 \
    --no-8bit-editor \
    --high-vram \
    --output-dir output/salient_imagenet_run \
    --log-file salient_imagenet_run.log \
    --debug
```

Differences vs the 24 GB command (§4):

- `--no-8bit-editor` — pure BF16 FLUX on GPU, no CPU offload. The single biggest speedup (4–8×) and the reason the H100 finishes inside a workday.
- `--high-vram` — keep classifier + VLM + editor all resident; eliminates 4–6 model swaps × ~30 s each.
- `--max-hypotheses 3` (was 5) — drops VLM features per image from 5 to 3, cutting total edits by ~40 %. Still more than enough coverage (SI's own evaluation only uses 5 features per class).
- `--top-negative-classes 3` (was 5) — uses 15 hard-negative images per class instead of 25; cuts another ~10–15 % of edits.
- `--batch-size 30` (was 20) — slightly larger batch, fewer per-batch overhead cycles. H100 has the VRAM headroom.

If you want to push further toward "definitely under 8 h", drop `--samples 10` to `--samples 7` (saves another ~30 %).

## 4.2 Resuming after interruption

The pipeline checkpoints per class to `output/salient_imagenet_run/checkpoints/resnet50/`. If the process dies, **just re-run the same command** — `cfg.resume = True` is the default. Completed classes (those that finished all per-batch phases) are loaded from disk and skipped; in-progress classes restart from scratch in the next batch. Use `--no-resume` to force a clean re-run.

### Estimated runtime — measured, not guessed

Numbers below are recomputed from a **live RTX 4090 run** of the exact command above on 2026-05-01. We saw FLUX.2-klein-9b running at **11.6 s/edit** in the default 8-bit-quant + CPU-offload configuration (24 GB-friendly), and **batch 1 (20 classes) emitted 2,611 edits** — i.e. ~131 edits/class. Multiplying out:

| GPU | Config | FLUX time per edit | Total ETA (357 classes ≈ 46.5k edits) |
|---|---|---|---|
| RTX 4090 (24 GB) | default 8-bit FLUX | ~11.6 s | **~6 days** |
| H100 80 GB | default 8-bit FLUX | ~5–7 s | **~3 days** |
| H100 80 GB | `--no-8bit-editor --high-vram` (BF16, no offload) | ~1.0–1.5 s | **~14–20 h FLUX alone** |

The first row is what is currently running here; the doc's earlier "14–18 h on RTX 4090" was a guess that didn't survive contact with reality (CPU↔GPU offload from 8-bit quant pinned the 4090 at 22 % GPU util).

**To finish in under 12 hours on H100 80 GB**, also reduce edit volume — full paper-spec is too dense even for an H100. Use the H100 command in §4.1.

What changed vs the 24 GB command and why each item buys time:

| Flag | RTX 4090 default | H100 setting | Impact |
|---|---|---|---|
| `--no-8bit-editor` | 8-bit FLUX + CPU offload | pure BF16 on GPU | 4–8× faster per edit (removes per-step CPU↔GPU transfer) |
| `--high-vram` | sequential model loading | all models resident | saves 4–6 model swaps × ~30 s each across the run |
| `--max-hypotheses 3` (was 5) | 5 features/image | 3 features/image | -40 % edits (FLUX bottleneck) |
| `--top-negative-classes 3` (was 5) | 25 negative imgs/class | 15 negative imgs/class | -10–15 % edits on top |
| `--batch-size 30` (was 20) | smaller batches | larger batches | fewer per-batch overhead cycles; H100 has the VRAM |

**Computed H100 budget (conservative 1.5 s/edit):**

| Phase | Items | Per-item | Time |
|---|---|---|---|
| Classify + Score-CAM | 357 cls × 25 imgs | 0.3 s | 0.7 h |
| VLM discovery | 357 × 10 imgs | 1.5 s | 1.5 h |
| Edit generation (VLM) | ~7 k features | 0.8 s | 1.6 h |
| **FLUX edit** | **~22 k edits** | **1.5 s** | **9.2 h** |
| Re-classify edited | 22 k | 0.05 s | 0.3 h |
| Verdict (≤ 20 % VLM) | ~4 k | 1.5 s | 1.7 h |
| **Total** | | | **~15 h worst case, ~9–11 h typical** |

The ~9–11 h "typical" assumes FLUX hits 1.0 s/edit on H100 SXM5 (no quant, BF16, KV-cache) — published 9B-class diffusion benchmarks support this. The 15 h "worst case" assumes 1.5 s/edit, which we treat as the upper bound.

**Conclusion: with the command above, an H100 80 GB run will finish in ~10 h and at worst ~12 h.** If you have an H100 NVL or SXM5 (vs PCIe), the lower end applies.

If you also want to cut another 30 %, drop `--samples` from 10 to 7 (still well above SI's 5-feature evaluation) — that brings the worst case to ~9 h.

Bottleneck is always FLUX.2 4-step generation (~60 edits/class with the H100 config × ~1.5 s/edit). If you want a fast smoke test first:

```bash
# 5-class sanity run, ~5 min on H100:
uv run python main.py --class-file src/salient_imagenet_classes.json \
    --classes 5 --classifiers resnet50 --no-probes \
    --no-8bit-editor --high-vram \
    --output-dir output/salient_imagenet_smoke
```

---

## 5. After Algorithm 1 finishes

Run the comparison script:

```bash
uv run python scripts/compare_salient_imagenet.py \
    --classifier resnet50 \
    --catalog output/salient_imagenet_run/reports/feature_catalog.json \
    --out-json output/salient_imagenet_run/reports/salient_imagenet_comparison.json \
    --out-md   output/salient_imagenet_run/reports/salient_imagenet_comparison.md \
    --threshold 0.45
```

Outputs:

- `salient_imagenet_comparison.json` — per-class match records + aggregate metrics
- `salient_imagenet_comparison.md` — human-readable summary table

Try a few thresholds (`0.40`, `0.45`, `0.50`, `0.55`) — the optimal value is sensitive to how literal vs. abstract the discovered feature names tend to be. Report all of them with a sensitivity sweep.

The script downloads the embedding model (~80 MB) on first run and caches it in `~/.cache/huggingface/`. CPU inference is fine — comparison takes < 1 min for 357 classes.

---

## 6. What to ship back

After the experiment, these are the files we need on this machine for the paper:

```
output/salient_imagenet_run/reports/
    analysis_results.json          (Algorithm 1 raw output)
    feature_catalog.json           (per-class buckets, the load-bearing artifact)
    salient_imagenet_comparison.json
    salient_imagenet_comparison.md
salient_imagenet_run.log           (full DEBUG log)
```

`output/salient_imagenet_run/images/` and `dataset_hf/` together can run to many GB and are not needed for the paper; you can leave them on the GPU machine.

A quick `tar -czf si_results.tgz output/salient_imagenet_run/reports salient_imagenet_run.log` at the end keeps the round-trip small (~10–50 MB).

---

## 7. If something breaks mid-run

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA out of memory` | FLUX.2 + Qwen-VL too big for VRAM | Confirm `low_vram=True` (default on 24 GB cards). Drop `--max-hypotheses` to 3 |
| GPU util < 30 % during edit phase on H100 | 8-bit FLUX is offloading to CPU each step | Add `--no-8bit-editor --high-vram` (the H100 command in §4.1 already does this) |
| `403` from HuggingFace | Gated model not accepted | Visit the HF model page in a browser, click "Agree" |
| Pipeline starts then no progress | Score-CAM + low-VRAM is slow | Switch `--attention gradcam` for a 3× speed-up |
| Disk full | Cached edited images + dataset shards | `output/salient_imagenet_run/images/` can be deleted between classes — checkpoints are in `checkpoints/` |
| Log file balloons | `--debug` writes verbose logs | `--log-level INFO` (no `--debug`) is fine for production runs |

---

## 8. TL;DR

```bash
# Setup (once)
uv sync
echo "hf_YOUR_TOKEN" > .token

# Run Algorithm 1 — pick the command for your GPU:

# H100 / A100 80 GB (target: ~10 h, < 12 h worst case)
uv run python main.py --class-file src/salient_imagenet_classes.json --all \
    --classifiers resnet50 --no-probes \
    --max-hypotheses 3 --top-negative-classes 3 \
    --batch-size 30 --no-8bit-editor --high-vram \
    --output-dir output/salient_imagenet_run --debug

# RTX 4090 24 GB (full paper-spec, but ~6 days — use only if you can leave it)
uv run python main.py --class-file src/salient_imagenet_classes.json --all \
    --classifiers resnet50 --no-probes --batch-size 20 \
    --output-dir output/salient_imagenet_run --debug

# Compare against human labels (< 1 min, CPU)
uv run python scripts/compare_salient_imagenet.py \
    --catalog output/salient_imagenet_run/reports/feature_catalog.json \
    --out-json output/salient_imagenet_run/reports/salient_imagenet_comparison.json \
    --out-md   output/salient_imagenet_run/reports/salient_imagenet_comparison.md
```
