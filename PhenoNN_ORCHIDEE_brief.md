# Brief de projet — Amélioration du LAI dans ORCHIDEE via ML (PhenoNN)

## Contexte du poste

CDD ingénieur ML/couplage numérique. Objectif : améliorer la simulation du **Leaf Area Index (LAI)** dans le modèle de surface **ORCHIDEE** en remplaçant le module process-based actuel (limité pour les écosystèmes tropicaux et semi-arides) par un modèle d'apprentissage automatique basé sur le repo [PhenoNN](https://github.com/kardaneh/PhenoNN), entraîné sur les données satellite **GEOV2** et validé contre le réseau **PhenoCam**.

Référence de l'offre : CNRS UAR636-MARCAS-014.

## Architecture actuelle du repo PhenoNN

```
phenonn/
├── models/
│   ├── fcn.py              # réseau fully-connected (baseline NN)
│   ├── linear_baseline.py  # modèle linéaire de référence
│   ├── rnn.py               # LSTM/GRU
│   ├── transformer.py       # Transformer v1
│   └── transformerbis.py    # Transformer v2 (version la plus aboutie, 17 Ko)
├── data/        # chargement et préparation des données
├── training/    # boucle d'entraînement, loss, hyperparamètres
├── prediction/  # inférence
├── utils/       # fonctions utilitaires
├── cli.py / __main__.py
```

Données d'exemple dans `example/` :
- `gcc_pred_test_GR_mfull.csv` — prédictions GCC (site "GR")
- `gcc_rcc_mins_site_veg.csv` — minima GCC/RCC par site et végétation
- `example/testdata/GR_bullshoals.csv` — données test du site PhenoCam **Bull Shoals** (Missouri)
- `example/lstm_models/` — modèles LSTM sauvegardés

Pas de `requirements.txt` ni `environment.yml` — packaging via `pyproject.toml` (installer avec `pip install -e .`).

## Sources de données

- **GCC/RCC (PhenoCam Network)** : [phenocam.nau.edu](https://phenocam.nau.edu/webcam/) — séries temporelles de "verdeur" (Green Chromatic Coordinate) et "rougeur" (Red Chromatic Coordinate) dérivées de photos de canopée. Site de test du repo : [bullshoals](https://phenocam.nau.edu/webcam/sites/bullshoals/).
- **Dataset consolidé** : [PhenoCam Dataset v2.0 (ORNL DAAC)](https://daac.ornl.gov/VEGETATION/guides/PhenoCam_V2.html), 393 sites Amérique du Nord + Europe.
- **GEOV2 (= CGLS LAI V2)** : produit satellite LAI, résolution 1 km / 10 jours, dérivé de SPOT/VEGETATION et PROBA-V. Point important : **GEOV2 est lui-même produit par un réseau de neurones**, donc PhenoNN apprend à reproduire la sortie d'un autre modèle ML (implication : plafond de précision atteignable, propagation d'erreurs à documenter).

## Priorités techniques, dans l'ordre

### 1. Comprendre et consolider le modèle ML existant
- Lire en priorité `phenonn/models/rnn.py` et `transformerbis.py` (version la plus aboutie).
- Comprendre `phenonn/training/` : fonction de perte, hyperparamètres, stratégie de validation croisée.
- Comprendre `phenonn/data/` : comment GCC/RCC/GEOV2 sont transformés en tenseurs d'entrée.
- Déterminer quelle architecture (FCN, linéaire, LSTM/GRU, Transformer, Transformerbis) est actuellement retenue comme meilleure, et sur quelle base (métriques).

### 2. Diagnostiquer les limites du module process-based ORCHIDEE actuel
- Zones tropicales humides : seuil maximal de LAI imposé par type de végétation (PFT) → sous-estimation de la variabilité du LAI au-delà de ce seuil.
- Zones semi-arides : couplage LAI / fraction de sol nu mal contraint par manque d'observations → biais sur les flux d'évapotranspiration.
- Objectif du ML : apprendre directement depuis les observations satellite plutôt que depuis une paramétrisation physique rigide.

### 3. Optimiser les architectures et l'entraînement
- Comparer/affiner LSTM, GRU, Transformer sur les données GEOV2.
- Valider avec des métriques standard : RMSE, R², biais.
- Validation croisée indépendante avec PhenoCam (jamais vu pendant l'entraînement GEOV2) pour tester la généralisation réelle du modèle, notamment fidélité du cycle saisonnier (dates de début/fin de saison de croissance).

### 4. Export du modèle PyTorch vers TorchScript
- Sauvegarder le modèle entraîné au format **TorchScript** (`.pt`), étape préalable obligatoire au couplage Fortran.

### 5. Couplage avec ORCHIDEE via FTorch
- Bibliothèque : [Cambridge-ICCS/FTorch](https://github.com/cambridge-ICCS/FTorch) — permet d'appeler un modèle PyTorch directement depuis du Fortran sans réécrire le modèle.
- Workflow standard :
  1. Entraîner et exporter le modèle PyTorch en TorchScript.
  2. Écrire le code Fortran utilisant les bindings FTorch (`torch_model_load`, `torch_tensor_from_array`, `torch_model_forward`).
  3. Compiler avec CMake en liant contre FTorch.
- Exemple minimal Fortran :
```fortran
call torch_tensor_from_array(input_tensors(1), in_data, tensor_layout, torch_kCPU)
call torch_model_load(torch_net, 'path/to/saved/model.pt')
call torch_model_forward(torch_net, input_tensors, output_tensors)
```
- Ressource pratique : [FTorch-workshop](https://github.com/Cambridge-ICCS/FTorch-workshop).
- Points de vigilance : compatibilité, stabilité numérique, performance du modèle couplé dans ORCHIDEE (écrit en FORTRAN90).

### 6. Runs ORCHIDEE comparatifs et évaluation d'impact
- Simuler avec l'ancien module LAI (process-based) et le nouveau module (ML) sur les mêmes conditions.
- Quantifier l'impact sur les flux de **carbone, eau, énergie**.
- Focus particulier sur les régions tropicales et semi-arides (là où le module actuel est le plus faible).

### 7. Documentation et valorisation
- Documenter code, architecture, hyperparamètres.
- Rédiger rapports techniques et contribuer à des publications scientifiques (rédaction en anglais).

## Environnement technique à préparer

- Python (numpy, pandas, xarray, PyTorch) + FORTRAN90.
- `conda create -n phenonn python=3.9` puis `pip install -e ".[ci,docs,dev]"` (Python ≥3.9 requis pour les extras docs/Sphinx).
- Accès HPC / environnement Linux pour faire tourner ORCHIDEE.
- FTorch à installer et tester séparément (voir workshop ci-dessus) avant toute tentative de couplage réel.

## Questions à poser au stagiaire sortant (par mail, avant l'arrivée)

1. Quelle architecture est actuellement retenue comme "meilleure" (FCN, linéaire, RNN, Transformer, Transformerbis) et sur quelles métriques (RMSE/R²/biais) cette décision repose-t-elle ?
2. Où sont stockées les données GEOV2 et PhenoCam complètes (au-delà des exemples du repo) — serveur du labo, cluster HPC, chemin partagé ?
3. Le travail d'export TorchScript et de test FTorch a-t-il déjà été commencé, même partiellement ?
4. Existe-t-il un rapport, une note technique ou un brouillon de publication décrivant les résultats obtenus jusqu'ici ?
5. Quel environnement HPC/modules est utilisé pour lancer ORCHIDEE, et existe-t-il un guide de démarrage ?

## Instructions pour l'agent de code

- Toujours vérifier la cohérence entre les scripts `training/` et les architectures dans `models/` avant de modifier un hyperparamètre.
- Ne pas modifier `transformerbis.py` sans comparer d'abord ses performances à `transformer.py` et `rnn.py` sur le même split de validation.
- Toute nouvelle fonctionnalité de préparation de données (GEOV2, PhenoCam) doit être testée sur le site `bullshoals` en premier (déjà présent dans `example/testdata/`) avant extension à d'autres sites.
- Prioriser la reproductibilité : fixer les seeds, versionner les configs d'entraînement, sauvegarder systématiquement en TorchScript après tout ré-entraînement jugé "final".
- Pour toute tâche liée à FTorch/Fortran, se référer d'abord aux exemples officiels du repo FTorch avant d'écrire du code de couplage custom.
