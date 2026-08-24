Scope and Architecture
======================

Scientific Scope
----------------

The active work targets global LAI simulation using GEOV2-derived observations,
independent seasonal validation against Phenocam, and later ORCHIDEE coupling
through FTorch. ORCHIDEE, Fortran, FTorch, production coupling, and impact
simulations are not present in this checkout.

Tensor Contract
---------------

Package inputs use ``(batch, features, sequence)``. Sequence models return
``(batch, output, sequence)``. The global path supplies 27 channels over 720
non-leap daily steps. ``Every10DaysWrapper`` selects the final year's days 5,
15, and 25, producing ``(batch, 1, 36)``.

Global Inputs
-------------

The 27 default channels are ordered as follows:

.. code-block:: text

   Tmin Tmax Tmean ssrd_sum strd_sum tp_sum
   VPD_max VPD_mean Rn_tot PET SMI
   co2
   pft1_frac ... pft15_frac

Raw physical units are the default. Optional normalization log-transforms
skewed channels and then applies train-primary z-score statistics. LAI targets
always remain in physical units.

Spatial and Temporal Independence
---------------------------------

The canonical selection stores explicit train, validation, test, and buffer
codes. Auxiliary sites belong to training only. Validation and test metrics use
primary sites only. Spatial transfer and temporal transfer should be evaluated
separately before one final locked-test evaluation.

Repository Boundaries
---------------------

Current package source lives under ``phenonn/``. Global data tooling lives under
``scripts/``. ``archive/`` contains historical scientific provenance and is not
imported or supported as executable package code.

The original project objective is recorded by CNRS offer UAR636-MARCAS-014:
https://emploi.cnrs.fr/Offres/CDD/UAR636-MARCAS-014/Default.aspx
