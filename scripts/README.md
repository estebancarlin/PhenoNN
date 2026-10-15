# scripts/ — repro entry points (not packaged)

| Script | What it does | Cost |
| --- | --- | --- |
| `smoke_train.sh` | Tiny repro: CPU, 2 epochs, few sites. Verifies the kept surface still works. | Minutes |
| `run_baseline.sh` | Small baseline: same trainer (`lstm` + `pft_mixing`), ~10x smoke, GPU-first with CPU fallback. | Minutes (GPU) / tens of minutes (CPU) |
| `phenonn-predict` | Legacy inference helper for pre-cleanup checkpoints. | — |

Usage:

```bash
DATA=/path/to/pixelset1 bash scripts/smoke_train.sh
DATA=/path/to/pixelset1 bash scripts/run_baseline.sh
DEVICE=cpu DATA=/path/to/pixelset1 bash scripts/run_baseline.sh   # force CPU
DEVICE=cuda DATA=/path/to/pixelset1 bash scripts/run_baseline.sh  # force GPU
```

Data lives outside the repo (`$DATA`, default `../pixelset1`).
Run outputs (checkpoints, logs) go under `runs/`.
