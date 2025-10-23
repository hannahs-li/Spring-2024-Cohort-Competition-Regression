## Regression Prediction Problem
Competition between spring 2024 data science cohort. Ranking: 25/2774

## Overview 
Predict Chicago Airbnb prices as of December 2023 based off of listing, host, and Airbnb characteristics in a Kaggle Competition for STAT 301-3.  

## Folders/directories
- `00_initial_setup/`: contains data set cleaning and subsequent splits and folds for `train_regression` data set and initial EDA of target variable 
- `01_model_building/`: contains r scripts for initial model building and tuning 
- `02_model_analysis/`: contains r scripts that combines model metrics for comparison and final submissions 
- `data/`: contains original `train_regression` data (derived from Airbnb) and it's codebook; cleaned, testing, training, and folds for `train_regressionn` dataset
- `figures_tables/`: contains all output of figures and tables
- `recipes/`: contains feature engineering and kitchen sink recipes 
- `results/`: contains fitted/tuned models, time logs, and metric evaluations of all models 
- `submissions/`: contains all model submissions made to Kaggle competition (models selected based off of MAE estimates derived from training set)

## Misc documents: 
- `.gitignore`: ignores large files
- `Li_Hannah_Regression_Problem.html`: final rendered html submission of project memo/short report
- `Li_Hannah_Regression_Problem.qmd`: final qmd submission of project memo/short report

