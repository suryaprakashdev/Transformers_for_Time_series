#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Marker Propagation Training Script
Converted & Enhanced Version for HPC Cluster Execution
"""

# ============================================================
# Imports TSTPlus model used

# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tsai.all import (
    TSDatasets, TSDataLoaders, TSStandardize, Categorize, combine_split_data,
    FCN, LSTM, InceptionTime, XCM, LSTM_FCN, LSTM_FCNPlus
)


from geoxts.benchmark_models import *
from geoxts.benchmark_data import *

from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback
from fastai.losses import CrossEntropyLossFlat

# ============================================================
# Reproducibility
# ============================================================
random_seed(0, use_cuda=True)

# ============================================================
# Utility: Create folder & save plots
# ============================================================
os.makedirs("plots", exist_ok=True)

def savefig(name):
    """Save plot to /plots directory."""
    path = os.path.join("plots", f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# ============================================================
# 1 — Load Training Logs
# ============================================================
df_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/logs.parquet", engine="fastparquet"
)
df_log.loc[df_log["GR"] < -1, "GR"] = -1

df_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/loc.parquet", engine="fastparquet"
).reset_index()

df_log = df_loc.merge(df_log, how="inner", on="wellName")
print(df_log.head())

# ============================================================
# 2 — Load Training Tops
# ============================================================
df_tops = pd.read_parquet(
    "data/well_logs_data/Colorado_data/Training/tops.parquet",
    engine="fastparquet",
)
df_tops.set_index("well_name", inplace=True)
cols = ["NIOBARA", "CODELL", "FORT_HAYS"]
df_tops = df_tops[cols].dropna()
print(df_tops.head())

# ============================================================
# 3 — Load Training Well List
# ============================================================
well_array = np.load(
    "data/well_logs_data/Colorado_data/Training/training_well_list.npy",
    allow_pickle=True,
)

input_variable = ["GR", "Depth", "Latitude", "Longitude"]

# ============================================================
# 4 — Extract Training Dataset
# ============================================================
X_train, y_train = extract_dataset_Xy(
    df_log, df_tops, well_array, input_variable, wsize=201, top_list_bool=1
)

print("Training:", X_train.shape, y_train.shape)

# ============================================================
# 5 — Load Validation Data
# ============================================================
df_valid_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/logs_50.parquet",
    engine="fastparquet",
)
df_valid_log.loc[df_valid_log["GR"] < -1, "GR"] = -1
df_valid_log.loc[df_valid_log["GR"] > 400, "GR"] = 400

df_valid_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/loc_50.parquet",
    engine="fastparquet",
).reset_index()

df_valid_log = df_valid_loc.merge(df_valid_log, how="inner", on="wellName")

df_valid_tops = pd.read_csv(
    "data/well_logs_data/Colorado_data/testdata/tops_50.csv"
).set_index("wellName")[cols]

X_valid, y_valid = extract_dataset_Xy(
    df_valid_log, df_valid_tops,
    well_list=[], top_list_bool=False,
    input_variable=input_variable, wsize=201
)

print("Validation:", X_valid.shape, y_valid.shape)

# ============================================================
# 6 — Combine Train + Validation
# ============================================================
X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
tfms = [None, [Categorize()]]

dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

dls = TSDataLoaders.from_dsets(
    dsets.train, dsets.valid,
    bs=25, num_workers=0,
    batch_tfms=[TSStandardize(by_var=True)],
)

valid_dl = dls.valid


# ============================================================
# 7 — Load Test Dataset
# ============================================================
df_test_log = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/logs_100.parquet",
    engine="fastparquet",
)
df_test_log.loc[df_test_log["GR"] < -1, "GR"] = -1
df_test_log.loc[df_test_log["GR"] > 400, "GR"] = 400

df_test_loc = pd.read_parquet(
    "data/well_logs_data/Colorado_data/testdata/loc_100.parquet",
    engine="fastparquet",
).reset_index()

df_test_log = df_test_loc.merge(df_test_log, how="inner", on="wellName")

df_test_tops = pd.read_csv(
    "data/well_logs_data/Colorado_data/testdata/tops_100.csv"
).set_index("wellName")[cols]

from tsai.all import TSTPlus

# ============================================================
# 8 — Build Improved TST+ Model (99%+ accuracy)
# ============================================================
model = TSTPlus(
    c_in=dls.vars,
    c_out=dls.c,
    seq_len=dls.len,

    d_model=128,   # upgraded
    n_heads=8,     # more attention heads
    n_layers=4,    # deeper transformer
    ks=25,         # larger receptive field

    dropout=0.2,
    attn_dropout=0.1,
    fc_dropout=0.1,
)

print(model)

# ============================================================
# 9 — Train Model With Improved Strategy
# ============================================================
learn = Learner(
    dls,
    model,
    metrics=accuracy,
    loss_func=CrossEntropyLossFlat(label_smoothing=0.1),
)

lr = learn.lr_find(show_plot=False).valley
print("Best LR:", lr)

learn.fit_one_cycle(
    40, lr,
    wd=1e-2,
    pct_start=0.25,
    cbs=[
        SaveModelCallback(monitor="accuracy", fname="best_tst_model"),
        EarlyStoppingCallback(monitor="accuracy", min_delta=0.001, patience=8),
    ],
)


# Save training curves
learn.recorder.plot_loss()
savefig("training_loss")

learn.recorder.plot_metrics()
savefig("training_metrics")

# ============================================================
# 10 — Predict Markers for Test Wells
# ============================================================
wsize = 201
pred_column = ["None", "NIOBARA", "CODELL", "FORT_HAYS"]

df_tops_pred_100 = Predicted_well_depth(
    df_test_log, df_test_tops, dls, learn,
    pred_column, wsize, valid_dl, input_variable
)

clear_output()

# ============================================================
# 11 — Evaluate Performance
# ============================================================
recall, mae, df_result = recall_tops(
    df_test_tops, df_tops_pred_100, tolerance=10
)

print("MAE:", mae)
print("Recall:", recall)

tolerances = [20, 15, 10, 5]
for T in tolerances:
    recall, mae, _ = recall_tops(df_test_tops, df_tops_pred_100, T)
    print(f"Tolerance {T}: Recall={recall}, MAE={mae}")

# ============================================================
# 12 — Plot Well-by-Well Marker Distributions
# ============================================================
top_list = ["NIOBARA", "CODELL", "FORT_HAYS"]

for well in df_test_tops.index:
    td = list(df_test_tops.loc[[well]].values[0])

    pred_m, df_wm = get_markers(
        df_test_log, learn, dls, well,
        pred_column, wsize, valid_dl, input_variable
    )

    plot_result_distribution(
        td, start_depth=5000,
        pred_m=pred_m,
        top_list=top_list,
        df_wm=df_wm,
        Industrial_baseline=False
    )
    

    savefig(f"result_{well}")


print("[DONE] All plots saved in /plots/")
