## 1. Setup and Requirements

*   Python environment: Ensure Python is installed on your system.

*   Libraries: Install required libraries

    * Install libraries using `pip`

            pip install numpy pandas torch


    * load custom libraries (dataloader_phenodata, lstm)

*   Hardware acceleration: Utilization of GPU for model training is
    recommended but not mandatory.

## 2. Preparing input data

DeepPhenoMem requires climatic data as input, including six dynamic
variables and two state variables.

* Dynamic Variables: Daily minimum temperature, daily maximum
    temperature, daily daylength, daily vapor pressure deficit, daily
    soil water availability, and daily shortwave radiation.

*   State Variables: Mean annual temperature and mean annual
    precipitation.

*   Input data should be provided in CSV format with appropriate column
    names and should cover two years of data to predict one year of GCC
    observations for the second year.

## 3. Running the model

*   Utilize the provided scripts:

    *  Use this script along with LSTM models to predict GCC based on your inputs.

           lstm_pred.py

    *  Supporting scripts for data
        preprocessing and LSTM model implementation.

           dataloader_phenodata.py

           lstm.py

*   Example usage:

    *   All required files are present in the "example" folder
        including:

        *   Testing data: 'GR_bullshoals.csv',  stored in
            '/example/testdata/GR_bullshoals.csv'

        *   Minimum GCC data: 'gcc_rcc_mins_site_veg.csv'

        *   LSTM models: modelname_pft_8f_modelnumber e.g. mfull_GR_8f_0 is the first Mfull model for GR

        *   Custom libraries: 'dataloader_phenodata.py' and 'lstm.py'

    *   Run the script:

             python lstm_pred.py full GR 4 'your_path/example'

        *   Here example is the input path, GR is the PFT, 4 is the
            batch size. Replace 'your_path/example' with the input
            path.

## 4. Output

*   The output consists of predicted GCC from all LSTM models for each PFT.

*   Compute the mean of the ensembled LSTM predictions to obtain the final prediction for GCC.
