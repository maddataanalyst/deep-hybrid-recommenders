# Pipeline experiment


## Overview

The main experimentation pipeline consists of a series of cross-validtion procedures for each algorithm:
1. **Simple collaborative filtering**;
2. **Deep collaborative filtering**;
3. **Hybrid recommender** (Deep colaborative filtering with additional feature information).

Within each cross-val procedure the following steps are performed:
1. The data is split into train-validation K-times
2. For each k-th approach, the model is fit on training and evaluated on validation data.
3. For each one of the K-attempts, the following metrics are collected:
   1. **MSE** - mean squared error on validtion set.
   2. **MAPE** - mean absolute percentage error on validation set 
   2. **MAE** - mean absolute error on validatiton set.
      
Metrics are then saved for analysis and comparison. 

After all cross validation procedures are done, overall analysis is performed, including:
1. **Kruskal Test** - Statistical test for presence of overall difference between models. Non parametric version of ANOVA.
2. **Pairwise non-parametric test** - Post-hoc pairwise testing for differences between each pair of models, with Bonferroni correction of p-values for multiple comparisons.

## Pipeline inputs

The input to each crossvalidtion pipeline consists of:
1. **Train X** data:
   1. X features (if used by the model)
   2. X user, item ids
2. **Train y** data
3. **Categorical data encoders** - their dimensionality and learned column unique values are used to build embeddings.
4. **User/item encoders** - their dimensionality and number of learned uniquue values are used to build embeddings.
5. **Model-specific parameters**.

## Pipeline outputs

Each cross-validation step outputs:
1. Set of K train metrics (MSE/MAPE/MAE)
2. Set of K validation metrics (MSE/MAPE/MAE)
3. Summary statistics (mean, standard deviation, etc.) for train and validation metrics.


Once all cross validation procedures are done, statistical comparisons are conducted and their output is saved. Specifically:
1. Boxplots for comparing metrics;
2. Pairwise comparison reports.