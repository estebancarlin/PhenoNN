# SpiritX (MESO IPSL-X) & HAL (MESO IPSL-G) Guide — PhenoNN

Practical guide for running PhenoNN on SpiritX (`spiritx2`, CPU) and
HAL (GPU). Copy this file to `~/PhenoNN/` on the server if useful.

## 1. Filesystems — where to put what

| Path | Quota | Backup / lifetime | Use for |
| --- | --- | --- | --- |
| `/home/$USER` (`$HOME`) | 32 GB, 400 000 files | Daily incremental | Code, venv, scripts, small configs |
| `/homedata/$USER` | 1 TB, 300 000 files | 15-day differential | Persistent data (GEOV2, ERA5, PFT, CO2), final checkpoints, results |
| `/scratchx/$USER` | 3 TB, 400 000 files (5 TB / 7 weeks on request) | **Purged after 6 months** | `uv` cache, `runs/`, checkpoints, staging, temp reorganisation |

Other clusters (`/spirit-home`, `/hal-home`, `/data`, `/scratchu`) are visible **read-only and slower** — do not write there.

PhenoNN mapping:

```bash
~/PhenoNN                  # git clone (code only)
~/PhenoNN/.venv            # python env (~6 GB, fits in $HOME quota)
/homedata/$USER/phenonn/data    # persistent datasets
/scratchx/$USER/phenonn/runs    # training outputs (checkout finalists to /homedata before purge)
/scratchx/$USER/uv-cache        # uv download cache
```

## 2. Python environment (uv, no sudo needed)

One-time setup per shell config:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export UV_CACHE_DIR=/scratchx/$USER/uv-cache' >> ~/.bashrc
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc   # cache and venv are on different filesystems
source ~/.bashrc
```

Clone and install (same as CI: `uv venv` + `uv pip install -e .[ci,dev]`):

```bash
git clone https://github.com/estebancarlin/PhenoNN.git ~/PhenoNN
cd ~/PhenoNN
mkdir -p /scratchx/$USER/uv-cache /scratchx/$USER/phenonn/runs /homedata/$USER/phenonn/data
ln -s /scratchx/$USER/phenonn/runs ~/PhenoNN/runs   # keep heavy outputs off $HOME quota
uv venv                          # Python 3.8 per .python-version
source .venv/bin/activate
uv pip install -e ".[ci,dev]"
uv pip install pytest            # NOT included in the dev extra
```

Verify:

```bash
.venv/bin/python tests/test_phenonn_installation.py   # run directly, not via pytest
.venv/bin/python -m unittest tests.test_transformer tests.test_rnn tests.test_fcn tests.test_utils tests.test_model_utils tests.test_transformerbis tests.test_evaluater tests.test_diagnostics
```

Notes:

- Login and compute nodes are **CPU-only** (no GPU partition on SpiritX);
  `torch` runs on CPU, which the smoke test confirms (`device: cpu`).
  The install still pulls CUDA wheels — harmless, just disk usage.
- Never compute on the login node beyond quick checks; use Slurm below.

## 3. Slurm on SpiritX — essentials

Discover state:

```bash
sinfo -s        # partitions, idle/mix/alloc nodes
squeue          # pending/running jobs (R, PD, CG, CD, CA)
slqueue         # friendlier squeue with pending reasons
check-cluster   # node states
```

Partitions (from `sinfo`):

| Partition | Nodes | Default use | Max time |
| --- | --- | --- | --- |
| `zen4` (default) | 64–96 cores, 248 GB (~3968 MB/cpu) | Default choice | 7 days |
| `zen16` | 32 cores, 496 GB (~15872 MB/cpu) | Only if you need > ~4 GB per cpu | 3 days |
| `fullnode` | whole node | Whole-node runs | 1 day |
| `jupyter` | — | JupyterHub | 12 h |
| `long` | — | Long single jobs | 7 days |

User limits (zen4/zen16): internal cpu=96 / mem=256 GB / 64 jobs;
external cpu=48 / mem=128 GB / 32 jobs. Over-limit jobs stay `PENDING`
(`Resources`, `Priority`, `QOSMaxCpuPerUserLimit`, `QOSMaxMemoryPerUser`,
`QOSMaxJobPerUser`, `ReqNodeNotAvail`/`Reserv`).

**Interactive** (max 10 h, **one at a time per user** — release when done):

```bash
srun --pty --x11 bash                                   # 1 core, 1 h, zen4, ~4 GB
srun --pty --x11 --mem 6G --time 2:00:00 bash           # 2 h, 6 GB (gets 2 cpus on zen4)
srun --pty --x11 --partition zen16 --mem 6G --time 2:00:00 bash  # same mem, 1 cpu
```

**Batch** (`sbatch script.sh`, directives as `#SBATCH` comments;
samples in `/net/nfs/tools/meso-u20/batch-samples`):

```bash
#!/bin/bash
#SBATCH --partition=zen4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratchx/%u/phenonn/slurm-%j.log
set -e
cd ~/PhenoNN && source .venv/bin/activate
python -m pytest tests/ --ignore=tests/test_phenonn_installation.py -q
```

Multithreaded Python: size pools from the allocation, not the machine:

```python
import os
n = int(os.getenv("SLURM_CPUS_PER_TASK", "1"))  # not os.cpu_count()
```

**Job arrays** (e.g. one job per seed 17 / 42 / 73, or per year):

```bash
sbatch --array=0-2 --ntasks=1 --cpus-per-task=4 --mem=16G --time=06:00:00 train_array.sh
# inside: SEEDS=(17 42 73); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
```

**Chaining** (resubmit at end of job, start next only on success):

```bash
sbatch --dependency=afterok:${SLURM_JOBID} next_step.sh
```

**Monitoring**: `sstat -a -j <jobid>` (running), `seff <jobid>` /
`jobreports -c -l` (finished). SSH to a node only to monitor your own job
(`top`/`htop`, look at `RES`); anything else via ssh can get the account revoked.

## 4. PhenoNN runs

Tests on a compute node:

```bash
srun --pty --mem 16G --time 2:00:00 bash
cd ~/PhenoNN && source .venv/bin/activate
python -m pytest tests/ --ignore=tests/test_phenonn_installation.py -q
```

Training (outputs land on scratch via the `runs` symlink):

```bash
sbatch -p zen4 -N 1 -n 1 --cpus-per-task=8 --mem=32G --time=24:00:00 \
  --output=/scratchx/$USER/phenonn/slurm-%j.log --wrap="cd $HOME/PhenoNN && source .venv/bin/activate && phenonn train-global --era-dir /homedata/$USER/phenonn/data/era5 --target-dir /homedata/$USER/phenonn/data/targets --selection <selection.nc> --output-dir /scratchx/$USER/phenonn/runs/<exp> --experiment <exp> --type lstm --train-years 1993-2014 --validation-years 2015-2016"
```

After training: copy checkpoints worth keeping to
`/homedata/$USER/phenonn/` (scratch is purged at 6 months).

## 6. HAL (MESO IPSL-G) — GPU runs

SpiritX is CPU-only: use HAL for GPU training. HAL has **no backup** —
copy results to a CPU cluster home or your PC when done.
Source: <https://documentations.ipsl.fr/spirit/hal_gpu_cluster/>.

### Access

- Head node: `hal.ipsl.fr`, SSH key only (ED25519 or RSA-4096, no password).
  Head node = short admin commands only, no scientific software installed.
- Compute nodes `hal[1-10]` are reachable only through a running Slurm job
  (no direct SSH).

### Hardware (Guyancourt, fast SSD storage)

| Nodes | CPU / RAM | GPUs |
| --- | --- | --- |
| hal1–hal3 | 16 cores Xeon Silver 4215, 64 GB | 2× RTX 2080 Ti 11 GB (Turing) |
| hal4 | 32 cores Xeon Silver 4215, 128 GB | 2× RTX 2080 Ti 11 GB (Turing) |
| hal5–hal6 | 40 cores Xeon Silver 4210R, 128 GB | 2× RTX A5000 24 GB (Ampere) |
| hal7 | 12 cores Xeon w3-2425, 128 GB | 1× RTX 4090 24 GB (Ada) |
| hal8–hal10 | 14 cores i5-14600KF, 64 GB | 1× RTX 4090 24 GB (Ada) |

For PhenoNN sizes prefer 24 GB VRAM nodes (hal5+); the 11 GB 2080 Ti
nodes are fine for tests and small configs.

### Slurm on HAL — partitions, limits, GPU request

- Partitions are **automatic**: `batch` for `sbatch` (default 1 h, max 72 h),
  `interactive` for `srun` (default 1 h, max 10 h). No `-p` needed.
- **Default GPU = 0: always request GPUs explicitly** with `--gpus`.
  Per-job hardware cap: 2 GPUs on hal1–6, 1 GPU on hal7–10.
- Session caps: max 1 node; internal users 2 GPUs (same node) and
  2 running jobs; external users 1 GPU and 1 running job.
- Arch selection: `--gpus=<ada|ampere|turing>:<n>`
  (e.g. `--gpus=ampere:1`). Unspecified = Ada first, then Ampere, then Turing.
- Our pip torch (2.4.1 + cu12) supports all three archs, so no module needed.
  Alternative: `module load pytorch/2.10.0` (check the
  [compatibility table](https://documentations.ipsl.fr/spirit/hal_gpu_cluster/ai_software.html)
  for module-vs-arch support).

```bash
srun --time='4:00:00' --gpus=1 --pty bash        # 1 GPU, interactive
srun --time='4:00:00' --gpus=ampere:1 --pty bash # force Ampere (24 GB)
nvidia-smi
```

Free stuck sessions: `scancel -u $USER` on the head node.

### Working spaces on HAL

| Path on HAL | Use for |
| --- | --- |
| `/home/$USER` (fast SSD, 20 GB, no file limit) | Code + venv (tight: CUDA venv is ~8 GB) |
| `/net/nfs/ssd1|$2|$3/$USER` (fast shared SSD, **create your dir**) | `uv` cache, datasets staged in, `runs/`, temp |
| `/bdd`, `/data/$USER`, `/scratchu/$USER`, `/scratchx/$USER`, `/ciclad-home`, `/climserv-home` | **Read-only**, slower (other sites over network) — copy in, never write |

Tip from the docs: data hosted at Jussieu/Polytechnique is read over the
network — always copy datasets to `ssd1/2/3` before training.

Setup (same `uv` flow, cache on SSD):

```bash
mkdir -p /net/nfs/ssd1/$USER/uv-cache /net/nfs/ssd1/$USER/phenonn/runs
git clone https://github.com/estebancarlin/PhenoNN.git ~/PhenoNN
cd ~/PhenoNN
ln -s /net/nfs/ssd1/$USER/phenonn/runs ~/PhenoNN/runs
export UV_CACHE_DIR=/net/nfs/ssd1/$USER/uv-cache UV_LINK_MODE=copy
uv venv && source .venv/bin/activate
uv pip install -e ".[ci,dev]" && uv pip install pytest
echo 'export UV_CACHE_DIR=/net/nfs/ssd1/$USER/uv-cache' >> ~/.bashrc
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
```

Stage data, then GPU shell + check:

```bash
cp -r /homedata/$USER/phenonn/data /net/nfs/ssd1/$USER/phenonn/data
srun --time='4:00:00' --gpus=1 --pty bash
cd ~/PhenoNN && source .venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

GPU training example (partition `batch` is automatic):

```bash
sbatch --gpus=1 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  --output=/net/nfs/ssd1/$USER/phenonn/slurm-%j.log --wrap="cd $HOME/PhenoNN && source .venv/bin/activate && phenonn train-global --era-dir /net/nfs/ssd1/$USER/phenonn/data/era5 --target-dir /net/nfs/ssd1/$USER/phenonn/data/targets --selection <selection.nc> --output-dir /net/nfs/ssd1/$USER/phenonn/runs/<exp> --experiment <exp> --type lstm --train-years 1993-2014 --validation-years 2015-2016"
```

Array over seeds (PhenoNN protocol: 17 / 42 / 73):

```bash
sbatch --array=0-2 --gpus=1 --cpus-per-task=8 --mem=64G --time=24:00:00 run_seed.sh
# in run_seed.sh: SEEDS=(17 42 73); SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
```

Extras: Jupyter via `/net/nfs/tools/bin/jupytercluster.sh 'pytorch/2.10.0'`
(+ TensorBoard with `-t <logdir>`); parallel HPO via Optuna
([doc](https://documentations.ipsl.fr/spirit/hal_gpu_cluster/optuna.html));
missing Python packages → extend the AI module or file a GitLab issue
(AI modules refresh each Jan–Feb).

After the run, copy survivors back (no backup on HAL):

```bash
# from spiritx2 (HAL filesystems are readable cross-cluster, slower):
cp -r /net/nfs/ssd1/$USER/phenonn/runs/<exp>/best_model.pth /homedata/$USER/phenonn/
# or scp/rsync hal → spiritx2
```

## 7. Pitfalls checklist

- Disk quota exceeded during install → `UV_CACHE_DIR` on scratch (done in §2).
- `runs/`, `data/`, caches in `$HOME` → quota is only 32 GB; use the symlink and `/homedata`.
- File-count quotas (300–400 k files): avoid exploding many small files; prefer NetCDF/zipped artifacts.
- Do not leave interactive `srun` shells idle; do not compute on login nodes.
- Multinode runs are not allowed on zen4/zen16 — stay single-node.
