# Experimental protocol: PhenoNN "memory" angle

Overall objective: **show that PhenoNN, trained on satellite LAI, learns a weather->phenology
memory (lagged / legacy effects) that process-based models miss**, quantify it
(length / shape / strength, SAM framework [28 Ogle]), and reproduce two key results:
the **drought legacy on the following year's SOS** [22 Liu Y.] and the **delayed effect of a
warm spring on summer LAI** [02 Buermann].

All paths below are relative to the repository root. The analysis code is under
`analysis/`, not inside the `phenonn/` package; `phenonn/` itself is not modified.

## 0. PhenoNN technical background (design constraints)
- **Input**: a **720-day** window = concat(year Y-1, year Y), the last 720 days
  (~ 10 Jan Y-1 -> 31 Dec Y). Daily ERA5 features (Tmin/Tmax/Tmean, ssrd_sum, strd_sum,
  tp_sum, VPD_max/mean, Rn_tot, PET) + daylength + SMI + CO2
  (**13 non-PFT channels total**) + **15 PFT fractions** = **28 channels**.
  Features are z-scored (log1p on heavy tails).
- **Output**: **36 dekadal LAI values for year Y** (predicted in one pass).
- Memory consequence: a perturbation in **Y-1** is **before** the SOS of Y
  (~ dekad 10-12, early April) -> an inter-annual legacy is **causal** and testable.
- **Intra-window causality**: the output for Y is produced from the WHOLE window,
  including the summer of Y (after the SOS of Y). Therefore:
  - For the **legacy from Y-1** (experiment M3), the perturbation is entirely before SOS(Y)
    -> **clean**.
  - For the **sensitivity analysis** (M1), interpret causally only the **lags earlier than
    the target dekad**, and prefer causal architectures (`bitransformer_v2`/`attnlstm`,
    `causal=True`).
- **Structural limitation**: PhenoNN only sees **weather** -> it captures **exogenous**
  memory (Ogle), **not** endogenous memory (NSC reserves, stress imprint [49, 50]).
  Measured effects are therefore a **lower bound**.
  Variant to test: add a **soil-moisture proxy** as an extra input channel to strengthen
  the memory pathway (not implemented yet).

## Shared toolkit (implement once under `analysis/memory_analysis/`)
1. **Phenology metrics** from the LAI curve (36 dekads -> daily interpolation with a
   double-logistic fit; add an `interpolate_lai_daily.py` helper), using the **threshold**
   method [D Bornez]:
   - amplitude A = max - min over the year; **SOS = first day with LAI > min + 0.30*A**
     (rising branch), **EOS = last day with LAI > min + 0.40*A** (falling branch),
     **LOS = EOS - SOS**, **peak** (day of maximum), **integral LAI** (seasonal integral,
     productivity proxy).
2. **Counterfactual inference runner**: load a checkpoint, edit the **raw weather window**
   (before normalization) over a date/variable range, then predict. Non-invasive mechanism
   modelled on `analysis/feature_ablation/train_ablate.py` (forward pre-hook, no change to
   `phenonn/`). This is the same approach as `analysis/perturbation/`.
3. **Drought operator** `D(window; time_window, severity s)`:
   `tp_sum *= (1-s)` over the targeted days (+ optional `VPD_max/mean *= 1+s'`,
   `PET *= 1+s'`).
4. **Stratifiers**: PFT (dominant fraction), **aridity** WI = sum(tp_sum) / sum(PET)
   (threshold 0.65, dryland/humid [21]), latitude.
5. **Contrast baselines**: (a) **XGBoost** weather->LAI without long memory (concurrent
   features only), (b) a simple **process/GDD** model. These show that the memory is
   specific to PhenoNN. Both are to be added in this study.

Data: use the **validation years** (held out from training) and `selected_pixels` for
honest evaluation.

---

## M1: Memory kernel (SAM framework [28]): *foundation*
**Hypothesis**: the sensitivity of LAI in dekad d to a weather perturbation at lag j
decreases with j, with a **non-zero memory length** (> a few weeks) and identifiable
**lags**.

**Method**:
- Target = a dekad d (e.g. SOS, peak, EOS dekad).
- **Autograd Jacobian**: `dLAI_d / d(feature v, day t)` over the whole 720-day window
  (one backward pass). Robust alternative: finite differences (+1 z-score perturbation
  on a single day t).
- Sweep t -> curve **S_d,v(lag)**; aggregate over N sites x years, **by PFT** and
  **by variable v**.

**Metrics**:
- **Length** L = lag beyond which |S| < 5 % of its maximum.
- **Shape**: peak positions (instantaneous ~lag 0 vs seasonal lags ~30/90/365 days).
- **Strength**: sum(|S|) over "memory" lags (> 30 days) / sum(|S|) total = **memory fraction**.

**Success criteria**: memory fraction clearly > 0; a cleanly decaying kernel;
**longer memory in drylands** (map of L, original figure).

---

## M2: Instantaneous vs memory decomposition (skill gain)
**Hypothesis**: lengthening the window / keeping the past improves prediction (useful
memory), especially **inter-annually** (cf. +18-28 % variance in Ogle).

**Method (2 variants)**:
- **A. Window truncation** (retraining): train PhenoNN on 720 / 365 / 180 / 90 days ->
  **R2 gain vs length** = memory contribution. Expensive but clean.
- **B. Inference-time masking** (no retraining): replace the **earlier** part of the window
  (> 90 days before each dekad) with the site **climatology** -> skill drop = memory
  contribution.

**Metrics**: Delta R2 **global / site-year / inter-annual** between full and ablated memory;
Delta RMSE on SOS/EOS.

**Success criteria**: memory Delta R2 > 0 (aim for a few points); **inter-annual** skill is the
most degraded by memory ablation (expected signature).

---

## M3: Drought legacy on the following year's SOS ([22]): *flagship experiment*
**Hypothesis**: a drought in Y-1 **delays the predicted SOS of year Y** (obs. +1.2 to
+2.3 days), more strongly in drylands, monotonically with severity: **where process models
do not**.

**Method**:
1. **Baseline**: predict year Y with observed weather -> SOS_base(Y), over N sites x years.
2. **Counterfactual**: apply `D(*, JJA of Y-1, s)` (deficit >= 2 consecutive months, as in
   [22]), unchanged Y -> SOS_drought(Y). **Delta SOS = SOS_drought - SOS_base**.
3. **Stratify** by aridity (WI) and PFT; **dose-response** over s in {0.2, 0.4, 0.6, 0.8}.
4. **3 drought types [22]** by varying how long the deficit persists into the **spring of
   Y** (type 1 = deficit not resolved by spring Y -> maximum delay; type 3 = resolved early
   -> weak effect).
5. **Model contrast**: same counterfactual on XGBoost / GDD -> should show **~0 delay**.

**Metrics**: Delta SOS distribution (mean, % delayed pixels), by WI/PFT; Delta SOS vs severity slope;
Delta SOS(type1) vs Delta SOS(type3).

**Success criteria**: **Delta SOS > 0** (order ~1-3 days); **stronger in drylands**; **monotone**
in severity; **type1 > type3**; **process baseline ~ 0** -> *PhenoNN captures the missing
exogenous legacy*.

**Pitfalls**: the effect is limited to the exogenous channel (inherited weather) -
magnitude is a lower bound; test the **+ SM proxy** variant to strengthen it. Check that
the Y-1 deficit is inside the window (it is).

---

## M4: Delayed effect of a warm spring on summer LAI ([02])
**Hypothesis**: a warm spring produces a **negative delayed effect** on summer LAI through
water stress, over ~10-15 % of the area (obs.), **unlike DGVMs** (which mostly predict a
positive effect).

**Method**:
- **A. Partial correlation** (direct replication of [02]) on **predicted LAI**: over the
  period, partial correlation spring T <-> summer LAI (controlling for concurrent summer
  climate) per pixel -> **map of the legacy sign** (+/-/0). Compare with the observed
  ~15 % negative / ~5 % positive.
- **B. Counterfactual**: `+Delta T` (e.g. +2 C) over **MAM of Y only** -> summer LAI trajectory
  (JJA) of Y; measure the **late anomaly**.

**Metrics**: % pixels with a negative legacy; dependence of the sign on
**precipitation/altitude** (their random forest); JJA LAI anomaly.

**Success criteria**: regions with a **negative legacy** (~10-15 %), concentrated in
**water-limited / high-altitude** areas; sign **driven by precipitation**.

---

## M5 (optional): Global map of memory length
From M1: map **L** per pixel/PFT -> drylands have long memory. **Cover figure**.

---

## Suggested order & dependencies
1. **Toolkit** (phenology metrics + counterfactual runner + drought operator).
2. **M1** (kernel) -> validates that memory exists and characterizes it (cheap, autograd).
3. **M3** (drought legacy) -> flagship result, relies on the toolkit.
4. **M2** (skill gain) and **M4** (spring) -> strengthen and generalize.
5. **M5** (map) -> final figure.

## Expected deliverables
- One module under `analysis/memory_analysis/` (phenology extraction, counterfactual runner,
  drought operator).
- Figures: kernel S_d(lag) by PFT; memory-length map; Delta SOS vs severity/aridity; map of the
  spring-legacy sign; instant vs memory Delta R2 table.
- Systematic comparison with >= 1 baseline without long memory (XGBoost/GDD).

## Scope / limitations to state in the paper
- **Exogenous** memory only; no endogenous (NSC) memory -> magnitudes are lower bounds
  ([49, 50, 28]).
- Target is **LAI** (canopy structure), not GCC or carbon flux -> "phenological response".
- Architectures: prefer `causal=True` for sensitivity interpretation.
- Scale dependence of metrics ([29]) + resolution mismatch 0.1 deg (weather) / 0.05 deg (target).
