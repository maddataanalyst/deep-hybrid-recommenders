# Pipeline modeling_prep

The goal of this pipeline is to prepare the reproducible experiment for all machine learning models. The pipeline will do the following:
1. Take the encoded X data:
    1. user ids
    2. item ids 
    3. item features extracted by encoders
3. Split the X data into train/validation sets and test sets.
4. Save splitted data as separate datasets.

Additional node in the pipeline gathers all metrics from all models and compares them.

## Overview

## Pipeline inputs

The pipeline takes as an input encoded User, Item ids and categocircal data.

## Pipeline outputs

The output from the pipelines consists of:
1. X train features
2. X train ids (user, item)
3. X test features
4. X test ids (user, item)
5. y train / test