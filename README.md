# BNetzA Data Analysis Repository

## Overview

This repository contains all the materials, data, and scripts used for analyzing data provided by the Federal Network Agency (BNetzA). The analysis primarily focuses on exploring, preprocessing, and modeling the data to extract meaningful insights.

## Repository Structure

- **`data/`**: Contains the original data files and supplementary geodata used for analysis.
  - `EVS4_20140118_dataV9.xlsx`: The primary dataset provided by BNetzA.
  - `germany-states.geojson`: Geospatial data for German federal states, used for mapping in the exploratory data analysis (EDA).

- **`Data_Preprocessing/`**: Includes notebooks and scripts for data preprocessing and exploratory analysis.
  - `data_preprocessing.py`: Contains the functions used for data cleaning, transformation, and preprocessing throughout the analysis.
  - `EDA_for_Data_Preprocessing.ipynb`: A notebook dedicated to initial exploratory data analysis focused on preparing the data for modeling.
  - `EDA_General.ipynb`: A broader exploratory analysis of the dataset, providing insights into the overall data distribution and characteristics.

- **`Figures/`**: This folder contains figures generated during the analysis, used in the term paper and reports.

- **`Modelling/`**: Contains the modeling scripts and notebooks used in the analysis.
  - `models.py`: Implementation of the various models used in the analysis.
  - `Models_Evaluation.ipynb`: The main notebook for evaluating models, which you can run to see the model performance and results. This notebook leverages functions implemented in `models.py`.
  - `cluster_based_modeling.ipynb`: An analysis focusing on clustering the Distribution System Operators (DSO) using the DBSCAN algorithm.
