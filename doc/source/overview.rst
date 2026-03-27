Overview
========

PhenoNN is a deep learning framework for predicting phenological metrics from climate data.
It uses LSTM (Long Short-Term Memory) networks to capture temporal dependencies in climate
time series and predict Green Chromatic Coordinate (GCC) for different plant functional types.

Architecture
------------

The PhenoNN architecture consists of:

1. **Data Preprocessing**: Climate data is standardized/normalized and formatted for LSTM input
2. **LSTM Network**: Multi-layer LSTM with configurable hidden size and dropout
3. **Training Pipeline**: Cross-validation, early stopping, and learning rate scheduling
4. **Prediction Pipeline**: Ensemble predictions from multiple trained models

Plant Functional Types
----------------------

PhenoNN supports three plant functional types:

- **DB (Deciduous Broadleaf)**: Deciduous broadleaf forests
- **EN (Evergreen Needleleaf)**: Evergreen needleleaf forests (coniferous)
- **GR (Grassland)**: Grasslands and herbaceous vegetation

Feature Sets
------------

The model can use different combinations of features:

- **6 features**: Dynamic variables only (temperature, precipitation, radiation, etc.)
- **8 features**: Dynamic + static climate (mean annual temperature and precipitation)
- **9 features**: Dynamic + static + snow cover
- **14 features**: All available variables (dynamic, static, soil properties)

Input Data Format
-----------------

Input data should be in CSV format with daily climate variables for two years to predict
one year of GCC observations. Required variables include:

**Dynamic Variables:**
- Daily minimum temperature (tmin)
- Daily maximum temperature (tmax)
- Daily daylength (daylength)
- Daily vapor pressure deficit (vpd)
- Daily soil water availability (swa)
- Daily
