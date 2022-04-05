# Temporally Sensitive Pretraining of Raw Feature Encoders

## Steps
1. Extract features from raw video using the (top level) `extract_features.py` script.
2. Pool for GVF from the extracted features using the `pool_for_gvf.py` script.
3. Use the GVF file in TSP script `tsp_train.py`.