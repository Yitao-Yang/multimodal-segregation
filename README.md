# multimodal-segregation
This repository contains the code implementing the methodology described in the paper titled "**Estimating social segregation as experienced through mobility**".

## Code Structure

The codebase is organized into four Python scripts reflecting the project workflow:

1.  `Data processing and fusion.py`: Handles initial data loading and preparation. Includes steps for inferring significant stays, home/work locations, socioeconomic status (SES) based on property data, training and applying a travel mode choice model, generating routes using an external API (Amap), and performing cross-data validation.
2.  `Measuring segregation.py`: Implements the core probabilistic framework for calculating the Probabilistic Segregation Measure (PSM) and the Multimodal Uniformity Index (MUI). It defines spatial and temporal units and provides functions for calculating these metrics across different transport modes and conducting sensitivity analyses on scale.
3.  `OLS models.py`: Contains functions to prepare data for and estimate Ordinary Least Squares (OLS) regression models. These models analyze the relationship between calculated PSM/MUI values and urban transport infrastructure characteristics at daily and hourly granularities.
4.  `Microscopic model of individual mobility.py`: Implements the agent-based model of individual mobility. It includes logic for model parameter calibration against empirical data and simulating the effects of various transport policy interventions on mode choices, travel costs, and resulting segregation patterns.

## Data Requirements

To run the scripts, several input data files are needed, expected in a `data_input_files/` subdirectory:

*   `mobile_pings.csv`: Raw mobile phone GPS trajectory data.
*   `poi_data.csv`: Points of Interest data (residential, commercial, public transport stations etc.).
*   `census_data.csv`: Aggregate census population data.
*   `lianjia_property_data.csv`: Residential property transaction data for SES inference.
*   `geolife_trips.csv`: Labeled trip data (e.g., from Geolife) for training the mode choice model.
*   `memda_speed_data.csv`: External traffic speed data for validation.
*   `city_boundary.shp`: Shapefile defining the study area boundary.
*   `region_to_spatial_unit_map.csv`: Mapping between spatial units (grids/segments) and larger regions.
*   **Intermediate/Derived Files:** Some scripts rely on outputs from previous stages. Examples include processed trip data with SES and mode probabilities, generated routes, hourly PSM/MUI metrics, and grid-level explanatory variables derived from infrastructure data.
*   **API Key:** An API key for a mapping service (like Amap) is required in `Data processing and fusion.py` for route generation.

*   ## Setup

1.  Clone the repository.
2.  Create a `data_input_files/` directory and place all necessary input data files within it.
3.  Install required Python packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `geopandas`, `shapely`, `statsmodels`, `requests`, `ast`.
4.  Replace `"AMAP_API_KEY"` placeholder in `Data processing and fusion.py` with your actual API key.

## How to Run

The scripts are intended to be run sequentially. Execute them from your terminal:

1.  `python "Data processing and fusion.py"`
2.  `python "Measuring segregation.py"`
3.  `python "OLS models.py"`
4.  `python "Microscopic model of individual mobility.py"`

Note that processing large-scale data can be computationally intensive and time-consuming, especially the route generation step relying on an external API. The calibration step in the microscopic model currently uses simplified calculations and a small grid search for demonstration; full-scale analysis may require more sophisticated optimization techniques.

## Outputs

The scripts output results primarily through standard logging messages printed to the console, detailing the progress, validation metrics, model summaries, calibrated parameters, and policy simulation outcomes (e.g., mode shares, cost changes, PSM changes). Some intermediate or final results could be added to be explicitly saved to files within the scripts.
