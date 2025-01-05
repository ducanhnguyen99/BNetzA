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


# Predicting DSO Costs in German Energy Distribution Networks

## Project Overview

This project explores the regulation of energy distribution networks in Germany, focusing on predicting the costs of Distribution System Operators (DSOs). Managed by the Bundesnetzagentur (BNetzA), efficiency benchmarking utilizes techniques like Data Envelopment Analysis (DEA) and Stochastic Frontier Analysis (SFA) to assess DSO performance. Accurate cost prediction models are essential for setting revenue caps that promote operational efficiency while ensuring fair pricing for consumers.

## Methodology

- **Data Analysis & Preprocessing:**  
  - **Dataset:** 194 network providers with 904 variables categorized into Capacity, Service, and Transport dimensions.
  - **Exploratory Data Analysis (EDA):** Identified skewed distributions, high sparsity, and multicollinearity among features.
  - **Data Cleaning:** Removed features with over 90% zero values and addressed skewness through logarithmic transformations.
  - **Feature Engineering:** Created multiple datasets with various transformations to capture nonlinear relationships and enhance model performance.

- **Modeling Approach:**  
  - Employed regression models including Lasso Regression, Linear Regression, Decision Tree Regression, and Random Forest.
  - Implemented a two-stage training process across multiple random seeds to ensure feature selection stability and model robustness.
  - Conducted hyperparameter tuning and evaluated models using metrics like RMSE, MAE, and MAPE.

## Key Findings

- **Model Performance:**  
  - **Lasso Regression** achieved the lowest prediction errors with a Testing MAPE of 0.126 and RMSE of 9.5 million, outperforming other models.
  - **Linear Regression** also demonstrated strong performance with a Testing MAPE of 0.138.
  - **Tree-Based Models** like Random Forest showed consistent feature importance but were less effective in capturing linear relationships compared to linear models.

- **Impact of Feature Selection:**  
  - **Key Predictors:** Energy losses (`yEnergy.losses.tot`) and energy delivered (`yEnergy.delivered.N1357.sum`) were identified as significant cost drivers.

- **Model Stability:**  
  - **Feature Stability:** Lasso Regression maintained stable feature selection across multiple iterations, enhancing model interpretability.
  - **Validation Challenges:** variability in performance metrics underscored the importance of using nested cross-validation to mitigate overfitting and ensure consistent model performance.

## Conclusion

The study demonstrates that linear models, particularly Lasso Regression with logarithmic transformations, are effective in predicting normalized total expenditures (cTOTEXn) for DSOs. These models not only provide accurate predictions but also offer interpretable insights into key cost drivers. Future research should incorporate more domain knowledge and explore advanced feature selection techniques to further enhance model reliability and accuracy.


