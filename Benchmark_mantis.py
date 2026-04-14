#!/usr/bin/env python
# -*- coding: utf-8 -*-



# ==========================================================
# 1. Imports
# ==========================================================

import pandas as pd
import numpy as np
import time
from IPython.display import clear_output

# tsai time series library
from tsai.all import (
    TSDatasets,
    TSDataLoaders,
    TSStandardize,
    Categorize,
    combine_split_data,
)

# benchmark dataset helpers
from geoxts.benchmark_data import *
from geoxts.benchmark_models import *

# models used for comparison
from tsai.all import (
    FCN,
    LSTM,
    InceptionTime,
    XceptionTime,
    XCM,
    LSTM_FCN,
    LSTM_FCNPlus,
)

# fastai
from fastai.learner import Learner
from fastai.metrics import accuracy

import torch
import torch.nn as nn


# ==========================================================
# 2. Reproducibility
# ==========================================================

random_seed(0, use_cuda=True)


# ==========================================================
# 3. Load Training Data
# ==========================================================

print("Loading training logs...")

df_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/logs.parquet",
    engine="fastparquet"
)

# clean GR values
df_log.loc[df_log["GR"] < -1, "GR"] = -1

df_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/loc.parquet",
    engine="fastparquet"
)

df_loc = df_loc.reset_index()

# merge location info
df_log = df_loc.merge(df_log, how="inner", on="wellName")

print(df_log.head())


# ==========================================================
# 4. Load Formation Tops
# ==========================================================

df_tops = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/tops.parquet",
    engine="fastparquet"
)

df_tops.set_index("well_name", inplace=True)

cols = ["NIOBARA", "CODELL", "FORT_HAYS"]

df_tops = df_tops[cols]

df_tops.dropna(inplace=True)

print(df_tops.head())


# ==========================================================
# 5. Load Training Well List
# ==========================================================

well_array = np.load(
    "data/well_logs_data/Colorado_data/Training/training_well_list.npy",
    allow_pickle=True
)


# ==========================================================
# 6. Extract Training Dataset
# ==========================================================

input_variable = ["GR", "Depth", "Latitude", "Longitude"]

print("Extracting training windows...")

X_train, y_train = extract_dataset_Xy(
    df_log,
    df_tops,
    well_array,
    input_variable,
    wsize=201,
    top_list_bool=1
)

print("Training shape:", X_train.shape, y_train.shape)


# ==========================================================
# 7. Load Validation Data
# ==========================================================

df_valid_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/logs_50.parquet",
    engine="fastparquet"
)

df_valid_log.loc[df_valid_log["GR"] < -1, "GR"] = -1
df_valid_log.loc[df_valid_log["GR"] > 400, "GR"] = 400

df_valid_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/loc_50.parquet",
    engine="fastparquet"
)

df_valid_loc = df_valid_loc.reset_index()

df_valid_log = df_valid_loc.merge(
    df_valid_log,
    how="inner",
    on="wellName"
)

df_valid_tops = pd.read_csv(
    "data/well_logs_data/Colorado_data/testdata/tops_50.csv"
)

df_valid_tops = df_valid_tops.set_index("wellName")

df_valid_tops = df_valid_tops[cols]


# ==========================================================
# 8. Extract Validation Dataset
# ==========================================================

X_valid, y_valid = extract_dataset_Xy(
    df_valid_log,
    df_valid_tops,
    [],
    input_variable,
    wsize=201,
    top_list_bool=0
)

print("Validation shape:", X_valid.shape, y_valid.shape)


# ==========================================================
# 9. Combine Dataset
# ==========================================================

X, y, splits = combine_split_data(
    [X_train, X_valid],
    [y_train, y_valid]
)

tfms = [None, [Categorize()]]

dsets = TSDatasets(
    X,
    y,
    tfms=tfms,
    splits=splits,
    inplace=True
)

dls = TSDataLoaders.from_dsets(
    dsets.train,
    dsets.valid,
    bs=25,
    batch_tfms=[TSStandardize(by_var=True)],
    num_workers=0
)

valid_dl = dls.valid

print("DataLoaders created.")


# ==========================================================
# 10. Load Test Dataset
# ==========================================================

df_test_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/logs_100.parquet",
    engine="fastparquet"
)

df_test_log.loc[df_test_log["GR"] < -1, "GR"] = -1
df_test_log.loc[df_test_log["GR"] > 400, "GR"] = 400

df_test_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/loc_100.parquet",
    engine="fastparquet"
)

df_test_loc = df_test_loc.reset_index()

df_test_log = df_test_loc.merge(
    df_test_log,
    how="inner",
    on="wellName"
)

df_test_tops = pd.read_csv(
    "data/well_logs_data/Colorado_data/testdata/tops_100.csv"
)

df_test_tops = df_test_tops.set_index("wellName")

df_test_tops = df_test_tops[cols]


# ==========================================================
# 11. MANTIS FOUNDATION MODEL
# ==========================================================

class MANTIS_TS(nn.Module):
    """
    Simplified MANTIS-style Transformer
    for Time-Series Classification
    """

    def __init__(self, c_in, c_out, seq_len):
        super().__init__()

        self.embedding = nn.Conv1d(
            c_in,
            128,
            kernel_size=5,
            padding=2
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(128, c_out)

    def forward(self, x):

        # input: (batch, channels, seq_len)

        x = self.embedding(x)

        # convert to transformer format
        x = x.permute(0, 2, 1)

        x = self.encoder(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x


# ==========================================================
# 12. Model List for Benchmark
# ==========================================================

archs = [
    (LSTM_FCN, {}),
    (LSTM_FCNPlus, {}),
    (XCM, {}),
    (FCN, {}),
    (InceptionTime, {}),
    (XceptionTime, {}),
    (LSTM, {"n_layers": 3, "bidirectional": False}),
    (LSTM, {"n_layers": 3, "bidirectional": True}),
    (MANTIS_TS, {})   # <-- foundation model
]


# ==========================================================
# 13. Training Benchmark
# ==========================================================

results = pd.DataFrame(
    columns=[
        "arch",
        "hyperparams",
        "total params",
        "train loss",
        "valid loss",
        "accuracy",
        "time"
    ]
)

result_recall = pd.DataFrame(
    columns=[
        "arch",
        "hyperparams",
        "mae",
        "recall",
        "Run time"
    ]
)

pred_column = ["None", "NIOBARA", "CODELL", "FORT_HAYS"]


for i, (arch, k) in enumerate(archs):

    model = build_model(arch, dls=dls, **k)

    print("Training:", model.__class__.__name__)

    learn = Learner(
        dls,
        model,
        metrics=accuracy
    )

    start = time.time()

    learn.fit_one_cycle(
        20,
        1e-3
    )

    elapsed = time.time() - start

    vals = learn.recorder.values[-1]

    wsize = 201

    df_tops_pred = Predicted_well_depth(
        df_test_log=df_test_log,
        df_test_tops=df_test_tops,
        dls=dls,
        learn=learn,
        pred_column=pred_column,
        wsize=wsize,
        valid_dl=valid_dl,
        input_variable=input_variable
    )

    recall, mae, df_result = recall_tops(
        df_test_tops,
        df_tops_pred,
        tolerance=10
    )

    results.loc[i] = [
        arch.__name__,
        k,
        count_parameters(model),
        vals[0],
        vals[1],
        vals[2],
        int(elapsed),
    ]

    result_recall.loc[i] = [
        arch.__name__,
        k,
        mae,
        recall,
        int(elapsed),
    ]

    results.sort_values(
        by="accuracy",
        ascending=False,
        inplace=True
    )

    result_recall.sort_values(
        by="recall",
        ascending=False,
        inplace=True
    )

    clear_output()

    print("Accuracy Benchmark")
    print(results)

    print("\nRecall Benchmark")
    print(result_recall)


# ==========================================================
# End of Script
# ==========================================================
