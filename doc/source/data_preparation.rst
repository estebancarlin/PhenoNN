Data Preparation
================

Input Data Format
-----------------

PhenoNN requires climate data in CSV format with daily time series. Each site should have its own CSV file.

File Naming Convention
----------------------

Data files should follow the naming pattern: `{PFT}_{site}.csv`

For example:
- `GR_bullshoals.csv`
- `DB_harvard.csv`
- `EN_nippon.csv`

CSV Structure
-------------

The CSV file should contain the following columns:

.. list-table:: Required Columns
   :header-rows: 1

   * - Column
     - Description
     - Units
   * - year
     - Year of observation
     - YYYY
   * - doy
     - Day of year
     - 1-365/366
   * - tmin
     - Daily minimum temperature
     - °C
   * - tmax
     - Daily maximum temperature
     - °C
   * - daylength
     - Daily daylength
     - hours
   * - vpd
     - Vapor pressure deficit
     - kPa
   * - swa
     - Soil water availability
     - mm
   * - radiation
     - Shortwave radiation
     - W/m²
   * - mat
     - Mean annual temperature
     - °C
   * - map
     - Mean annual precipitation
     - mm
   * - snow
     - Snow cover
     - mm
   * - sand
     - Soil sand content
     - %
   * - silt
     - Soil silt content
     - %
   * - clay
     - Soil clay content
     - %
   * - ph
     - Soil pH
     - pH units
   * - gcc
     - Green Chromatic Coordinate
     - unitless
   * - rcc
     - Red Chromatic Coordinate
     - unitless
   * - gcc_lowess
     - GCC with LOWESS smoothing
     - unitless
   * - rcc_lowess
     - RCC with LOWESS smoothing
     - unitless

Data Requirements
-----------------

- **Time span**: Minimum 2 years of data to predict 1 year of GCC
- **Missing values**: Should be handled before input (interpolation recommended)
- **Data range**: Should be within realistic bounds for each variable

Directory Structure
-------------------

For predictions, organize your data as:

.. code-block:: text

   your_data/
   ├── testdata/
   │   ├── GR_site1.csv
   │   ├── GR_site2.csv
   │   └── ...
   ├── lstm_models/
   │   ├── mfull_GR_8f_0
   │   ├── mfull_GR_8f_1
   │   └── ...
   └── gcc_rcc_mins_site_veg.csv

Minimum GCC File
----------------

The `gcc_rcc_mins_site_veg.csv` file should contain minimum GCC values for each site:

.. code-block:: text

   ,0,1,2,3,4
   GR_bullshoals,GR_bullshoals,GR,site,0.287,0.487
   DB_harvard,DB_harvard,DB,site,0.295,0.512
