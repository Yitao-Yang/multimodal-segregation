# multimodal-segregation
This repository contains the code implementing the methodology described in the paper titled "**Estimating social segregation as experienced through mobility**".

## Code Structure

The codebase is organized into four Python scripts reflecting the project workflow:

1.  `Data processing and fusion.py`: Handles initial data loading and preparation. Includes steps for inferring significant stays, home/work locations, socioeconomic status (SES) based on property data, training and applying a travel mode choice model, generating routes using an external API (Amap), and performing cross-data validation.
2.  `Measuring segregation.py`: Implements the core probabilistic framework for calculating the Probabilistic Segregation Measure (PSM) and the Multimodal Uniformity Index (MUI). It defines spatial and temporal units and provides functions for calculating these metrics across different transport modes and conducting sensitivity analyses on scale.
3.  `OLS models.py`: Contains functions to prepare data for and estimate Ordinary Least Squares (OLS) regression models. These models analyze the relationship between calculated PSM/MUI values and urban transport infrastructure characteristics at daily and hourly granularities.
4.  `Microscopic model of individual mobility.py`: Implements the agent-based discrete choice model of individual mobility. It includes logic for model parameter calibration against empirical data and simulating the effects of various transport policy interventions on mode choices, travel costs, and resulting segregation patterns.
