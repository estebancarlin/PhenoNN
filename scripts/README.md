# scripts/ — repro entry points (not packaged)

| Script | What it does | Cost |
| --- | --- | --- |
| `smoke_train.sh` | Tiny repro: CPU, 2 epochs, 8 sites/epoch, train 2000-2001 / val 2002. Verifies the kept surface still works after cleanup. | Minutes |
| `run_baseline.sh` | Small baseline: same trainer (`phenonn.training.train_full_ram`, `lstm` + `pft_mixing`), ~10x smoke (GPU-first with CPU fallback, 10 epochs, 32 sites/epoch, train 2000-2003 / val 2004). W1 baseline checkpoint. | Minutes (GPU) / tens of minutes (CPU) |
| `phenonn-predict` | Inference helper. | — |

Usage:

```bash
DATA=/data/ecarlin/pixelset1 bash scripts/smoke_train.sh
DATA=/data/ecarlin/pixelset1 bash scripts/run_baseline.sh
DEVICE=cpu DATA=/data/ecarlin/pixelset1 bash scripts/run_baseline.sh   # force CPU
DEVICE=cuda DATA=/data/ecarlin/pixelset1 bash scripts/run_baseline.sh  # force GPU
```

Data lives outside the repo (`$DATA`, default `../pixelset1`).
All run outputs (checkpoints, logs, CSVs) go under `runs/`, which is a
symlink to SSD storage (`/net/nfs/ssd2/ecarlin/runs` on the cluster).
Both scripts default to `$REPO/runs` (i.e. through the link); set
`RUNS_DIR=` only to bypass it.
Full SLURM / GPU runs live under `jobs/` (map only) and
`phenonn/analysis/*/jobs/`. Hand-run diagnostics live in
`phenonn/analysis/` (see its `README.md`), per `doc/DECISIONS.md`.
`smoke_train.sh` is CPU-only; `run_baseline.sh` is GPU-first
(batch 64, 4 workers, `--amp`) with automatic CPU fallback
(batch 16, 0 workers) when CUDA is unavailable or the GPU run fails
(known torch/CUDA kernel mismatch on the cluster).
