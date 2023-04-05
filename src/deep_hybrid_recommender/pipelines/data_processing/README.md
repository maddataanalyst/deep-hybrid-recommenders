# Pipeline data_processing

## Overview


This pipeline aims to prepare the data for processing by recommendation systems. The steps included in this pipeline are:
1. **First step**: building categorical data encoding - for "Style" column, which can have multiple values in the form of a dictionary. 
    > E.g.: `Style: {'Color': 'red', 'Size': 2oz, 'Scent': ' ...}`.
   1. Encoders are built for each possible nested Style attribute (Color/Size/Scent/etc.)
   2. Categorical data is numerically encoded for further processing by embeddings.
   3. Encoders and encoded data are saved for future processing.
2. **Second step**: building user id and item id encoders - user and item ids must be remapped to consecutive integer values starting from zero.


## Pipeline inputs

Input to the preprocessing data pipeline is a raw JSON file with AmazonReviews.

## Pipeline outputs

The pipeline saves:
1. Categorical encoders
2. User and item id encoders
3. Data with encoded values.