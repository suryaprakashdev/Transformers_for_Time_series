
import os
import time
import math
import pandas as pd
import numpy as np

# Safe display (works inside and outside IPython)
try:
    from IPython.display import display, clear_output
except Exception:
    def display(x):
        print(x)
    # Provide a noop clear_output for non-notebook runs
    def clear_output(wait=False):
        pass

# Create plot/table folders and define auto-save helpers
import matplotlib.pyplot as plt
os.makedirs("plots", exist_ok=True)
os.makedirs("tables", exist_ok=True)

plot_counter = 1
table_counter = 1

PLOT_PREFIX = "colorado_plot"
TABLE_PREFIX = "colorado_table"

def save_plot(name=None, dpi=300, bbox_inches='tight'):
    """
    Save the currently active matplotlib figure to plots/ folder.
    Call this after any function that generates plots (e.g., dls.show_batch()).
    """
    global plot_counter
    if name is None:
        filename = f"plots/{PLOT_PREFIX}_{plot_counter}.png"
    else:
        filename = f"plots/{name}.png"
    try:
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()
        print(f"[SAVED PLOT] {filename}")
    except Exception as e:
        print(f"[SAVE PLOT FAILED] {filename} -> {e}")
    plot_counter += 1
    return filename

def save_and_display_df(df, name=None):
    """
    Save dataframe to CSV (tables/) and display it (or print if not in notebook).
    """
    global table_counter
    if name is None:
        filename = f"tables/{TABLE_PREFIX}_{table_counter}.csv"
    else:
        # sanitize name
        safe_name = name.replace(" ", "_").replace("/", "_")
        filename = f"tables/{safe_name}.csv"
    try:
        # ensure dataframe-like
        if hasattr(df, "to_csv"):
            df.to_csv(filename, index=False)
            print(f"[SAVED TABLE] {filename}")
        else:
            # If not a dataframe, print and save repr
            with open(filename, "w", encoding="utf-8") as f:
                f.write(str(df))
            print(f"[SAVED TABLE AS TEXT] {filename}")
    except Exception as e:
        print(f"[SAVE TABLE FAILED] {filename} -> {e}")
    table_counter += 1
    try:
        display(df)
    except Exception:
        print(df)
    return filename

def auto_df(df, name=None):
    return save_and_display_df(df, name=name)


# -------------------------------
# Project code (adapted from notebook)
# -------------------------------

# Import domain-specific libs (assumes they are installed in your environment)
from geoxts.benchmark_data import *          # provides extract_dataset_Xy, Predicted_well_depth, recall_tops, random_seed, etc.
from tsai.all import TSDatasets, TSDataLoaders, TSStandardize, Categorize, combine_split_data
from tsai.all import FCN, LSTM, InceptionTime, XceptionTime, XCM, LSTM_FCN, LSTM_FCNPlus
from geoxts.benchmark_models import *        # provides build_model, count_parameters, and custom archs used below

# Set random seed (use_cuda True if you want to use GPU)
random_seed(0, use_cuda=True)

# --- Data loading & preprocessing ---
print("Loading logs data...")
df_log = pd.read_parquet('data/well_logs_data/Colorado_data/Training/logs.parquet', engine='fastparquet')
df_log.loc[df_log['GR'] < -1, 'GR'] = -1
df_loc = pd.read_parquet('data/well_logs_data/Colorado_data/Training/loc.parquet', engine='fastparquet')
df_loc = df_loc.reset_index()
df_log = df_loc.merge(df_log, how='inner', left_on='wellName', right_on='wellName')
print(df_log.head())

df_tops = pd.read_parquet('data/well_logs_data/Colorado_data/Training/tops.parquet', engine='fastparquet')
df_tops.set_index('well_name', inplace=True)
cols = ['NIOBARA','CODELL', 'FORT_HAYS']
df_tops = df_tops[cols]
df_tops.dropna(inplace=True)
print("Tops head:")
auto_df(df_tops.head(), name="training_tops_head")

# Load training well list
well_array = np.load('data/well_logs_data/Colorado_data/Training/training_well_list.npy', allow_pickle=True)

input_variable = ['GR', 'Depth','Latitude','Longitude']
# extract training data
X_train, y_train = extract_dataset_Xy(df_log, df_tops, well_array, input_variable, wsize=201, top_list_bool=1)
print("Training shapes:", X_train.shape, y_train.shape)

# Validation dataset
df_valid_log = pd.read_parquet('data/well_logs_data/Colorado_data/testdata/logs_50.parquet', engine='fastparquet')
df_valid_log.loc[df_valid_log['GR'] < -1, 'GR'] = -1
df_valid_log.loc[df_valid_log['GR'] > 400, 'GR'] = 400
df_valid_loc = pd.read_parquet('data/well_logs_data/Colorado_data/testdata/loc_50.parquet', engine='fastparquet')
df_valid_loc = df_valid_loc.reset_index()
df_valid_log = df_valid_loc.merge(df_valid_log, how='inner', left_on='wellName', right_on='wellName')
df_valid_tops = pd.read_csv('data/well_logs_data/Colorado_data/testdata/tops_50.csv')
df_valid_tops = df_valid_tops.set_index('wellName')
df_valid_tops = df_valid_tops[cols]

X_valid, y_valid = extract_dataset_Xy(df_valid_log, df_valid_tops, [], input_variable, wsize=201, top_list_bool=0)
print("Validation shapes:", X_valid.shape, y_valid.shape)

# Combine & create tsai datasets / dataloaders
X, y, splits = combine_split_data([X_train, X_valid], [y_train, y_valid])
tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=25, batch_tfms=[TSStandardize(by_var=True)], num_workers=0)

# Show batch (plot) and save it
try:
    dls.show_batch()
    # show_batch triggers matplotlib; save it
    save_plot("colorado_show_batch")
except Exception as e:
    print("[show_batch failed]", e)

xb, yb = dls.one_batch()
print("One batch labels example:", yb)

# Test dataset for recall calculation
df_test_log = pd.read_parquet('data/well_logs_data/Colorado_data/testdata/logs_100.parquet', engine='fastparquet')
df_test_log.loc[df_test_log['GR'] < -1, 'GR'] = -1
df_test_log.loc[df_test_log['GR'] > 400, 'GR'] = 400
df_test_loc = pd.read_parquet('data/well_logs_data/Colorado_data/testdata/loc_100.parquet', engine='fastparquet')
df_test_loc = df_test_loc.reset_index()
df_test_log = df_test_loc.merge(df_test_log, how='inner', left_on='wellName', right_on='wellName')
df_test_tops = pd.read_csv('data/well_logs_data/Colorado_data/testdata/tops_100.csv')
df_test_tops = df_test_tops.set_index('wellName')
df_test_tops = df_test_tops[cols]

# Architectures to evaluate (existing ones)
archs = [(LSTM_FCN, {}), (LSTM_FCNPlus, {}), (XCM, {}), (FCN, {}), (InceptionTime, {}), (XceptionTime, {}), 
         (LSTM, {'n_layers':3, 'bidirectional': False}), (LSTM, {'n_layers':3, 'bidirectional': True})]

from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback

# --- Experiment Loop 1 ---
results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
result_recall = pd.DataFrame(columns=['arch', 'hyperparams','mae', 'recall', 'Run time'])
pred_column = ['None','NIOBARA','CODELL', 'FORT_HAYS']

for i, (arch, k) in enumerate(archs):
    print(f"Building model for arch: {getattr(arch, '__name__', str(arch))} with params {k}")
    model = build_model(arch, dls=dls, **k)
    print("Model built:", model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy)
    start = time.time()
    learn.fit_one_cycle(20, 1e-3)
    elapsed = time.time() - start
    vals = learn.recorder.values[-1] if len(learn.recorder.values) > 0 else [None, None, None]
    wsize = 201
    try:
        df_tops_pred = Predicted_well_depth(df_test_log=df_test_log, df_test_tops=df_test_tops, dls=dls, learn=learn, pred_column=pred_column, wsize=wsize, valid_dl=dls.valid, input_variable=input_variable)
    except Exception as e:
        print("[Predicted_well_depth failed]", e)
        df_tops_pred = None
    rtime = time.time() - start
    try:
        recall, mae, df_result = recall_tops(df_test_tops, df_tops_pred, tolerance=10)
    except Exception as e:
        print("[recall_tops failed]", e)
        recall, mae, df_result = None, None, None

    results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed) if elapsed is not None else None]
    result_recall.loc[i] = [arch.__name__, k, mae, recall, int(rtime) if rtime is not None else None]

    # Sort and save intermediate results
    results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
    result_recall.sort_values(by='recall', ascending=False, kind='stable', ignore_index=True, inplace=True)
    clear_output(wait=True)

    # Save and display tables
    save_and_display_df(results, name=f"results_loop1_step_{i}")
    save_and_display_df(result_recall, name=f"result_recall_loop1_step_{i}")

# --- Additional architectures (made within benchmark_models) ---
archs_made = [(LSTM_2dCNN, {}), (LSTM_XCM, {}), (XCM_LSTM, {}), (LSTM_FCN_2dCNN, {})]

results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
result_recall = pd.DataFrame(columns=['arch', 'hyperparams','mae', 'recall', 'Run time'])
pred_column = ['None','NIOBARA','CODELL', 'FORT_HAYS']

for i, (arch, k) in enumerate(archs_made):
    print(f"Building model for arch: {getattr(arch, '__name__', str(arch))} with params {k}")
    model = build_model(arch, dls=dls, **k)
    print("Model built:", model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy)
    start = time.time()
    learn.fit_one_cycle(15, 1e-3)
    elapsed = time.time() - start
    vals = learn.recorder.values[-1] if len(learn.recorder.values) > 0 else [None, None, None]
    wsize = 201
    try:
        df_tops_pred = Predicted_well_depth(df_test_log=df_test_log, df_test_tops=df_test_tops, dls=dls, learn=learn, pred_column=pred_column, wsize=wsize, valid_dl=dls.valid, input_variable=input_variable)
    except Exception as e:
        print("[Predicted_well_depth failed]", e)
        df_tops_pred = None
    rtime = time.time() - start
    try:
        recall, mae, df_result = recall_tops(df_test_tops, df_tops_pred, tolerance=10)
    except Exception as e:
        print("[recall_tops failed]", e)
        recall, mae, df_result = None, None, None

    results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed) if elapsed is not None else None]
    result_recall.loc[i] = [arch.__name__, k, mae, recall, int(rtime) if rtime is not None else None]

    results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
    result_recall.sort_values(by='recall', ascending=False, kind='stable', ignore_index=True, inplace=True)
    clear_output(wait=True)
    save_and_display_df(results, name=f"results_archs_made_step_{i}")
    save_and_display_df(result_recall, name=f"result_recall_archs_made_step_{i}")

# --- Advanced training with lr_find and callbacks (original notebook had several loops like this) ---
results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
result_recall = pd.DataFrame(columns=['arch', 'hyperparams','mae', 'recall', 'Run time'])
pred_column = ['None','NIOBARA','CODELL', 'FORT_HAYS']

for i, (arch, k) in enumerate(archs):
    print(f"Building model for arch: {getattr(arch, '__name__', str(arch))} with params {k}")
    model = build_model(arch, dls=dls, **k)
    print("Model built:", model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy)
    start = time.time()
    try:
        lr_valley = learn.lr_find(show_plot=False)
        # lr_find returns a suggestion object in some fastai versions; guard for float
        if isinstance(lr_valley, dict) and "valley" in lr_valley:
            lr_val = lr_valley["valley"]
        elif hasattr(lr_valley, "suggestion"):
            lr_val = lr_valley.suggestion if hasattr(lr_valley, "suggestion") else 1e-3
        elif isinstance(lr_valley, float):
            lr_val = lr_valley
        else:
            lr_val = 1e-3
    except Exception as e:
        print("[lr_find failed]", e)
        lr_val = 1e-3

    try:
        learn.fit_one_cycle(20, lr_val, cbs=[SaveModelCallback(monitor='accuracy'), EarlyStoppingCallback(monitor='valid_loss', min_delta=0.01, patience=5)])
    except Exception as e:
        print("[fit_one_cycle with callbacks failed]", e)
        # fallback
        learn.fit_one_cycle(20, lr_val)

    elapsed = time.time() - start
    vals = learn.recorder.values[-1] if len(learn.recorder.values) > 0 else [None, None, None]
    wsize = 201
    try:
        df_tops_pred = Predicted_well_depth(df_test_log=df_test_log, df_test_tops=df_test_tops, dls=dls, learn=learn, pred_column=pred_column, wsize=wsize, valid_dl=dls.valid, input_variable=input_variable)
    except Exception as e:
        print("[Predicted_well_depth failed]", e)
        df_tops_pred = None
    rtime = time.time() - start
    try:
        recall, mae, df_result = recall_tops(df_test_tops, df_tops_pred, tolerance=10)
    except Exception as e:
        print("[recall_tops failed]", e)
        recall, mae, df_result = None, None, None

    results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed) if elapsed is not None else None]
    result_recall.loc[i] = [arch.__name__, k, mae, recall, int(rtime) if rtime is not None else None]

    results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
    result_recall.sort_values(by='recall', ascending=False, kind='stable', ignore_index=True, inplace=True)
    clear_output(wait=True)
    save_and_display_df(results, name=f"results_lrfind_loop_step_{i}")
    save_and_display_df(result_recall, name=f"result_recall_lrfind_loop_step_{i}")

# --- Another loop on archs_made (keeps original notebook behavior) ---
results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'train loss', 'valid loss', 'accuracy', 'time'])
result_recall = pd.DataFrame(columns=['arch', 'hyperparams','mae', 'recall', 'Run time'])
pred_column = ['None','NIOBARA','CODELL', 'FORT_HAYS']

for i, (arch, k) in enumerate(archs_made):
    print(f"Building model for arch: {getattr(arch, '__name__', str(arch))} with params {k}")
    model = build_model(arch, dls=dls, **k)
    print("Model built:", model.__class__.__name__)
    learn = Learner(dls, model, metrics=accuracy)
    start = time.time()
    learn.fit_one_cycle(15, 1e-3)
    elapsed = time.time() - start
    vals = learn.recorder.values[-1] if len(learn.recorder.values) > 0 else [None, None, None]
    wsize = 201
    try:
        df_tops_pred = Predicted_well_depth(df_test_log=df_test_log, df_test_tops=df_test_tops, dls=dls, learn=learn, pred_column=pred_column, wsize=wsize, valid_dl=dls.valid, input_variable=input_variable)
    except Exception as e:
        print("[Predicted_well_depth failed]", e)
        df_tops_pred = None
    rtime = time.time() - start
    try:
        recall, mae, df_result = recall_tops(df_test_tops, df_tops_pred, tolerance=10)
    except Exception as e:
        print("[recall_tops failed]", e)
        recall, mae, df_result = None, None, None

    results.loc[i] = [arch.__name__, k, count_parameters(model), vals[0], vals[1], vals[2], int(elapsed) if elapsed is not None else None]
    result_recall.loc[i] = [arch.__name__, k, mae, recall, int(rtime) if rtime is not None else None]

    results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
    result_recall.sort_values(by='recall', ascending=False, kind='stable', ignore_index=True, inplace=True)
    clear_output(wait=True)
    save_and_display_df(results, name=f"results_final_loop_step_{i}")
    save_and_display_df(result_recall, name=f"result_recall_final_loop_step_{i}")


# -------------------------------
# Append TSTPlus model (as requested)
# -------------------------------
import torch
import torch.nn as nn

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

class LinBnDrop(nn.Module):
    def __init__(self, in_features, out_features, p=0.1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, out_features)
        )
    def forward(self, x):
        return self.seq(x)

class _TSTEncoderLayer(nn.Module):
    def __init__(self, embed_dim=128, ff_dim=256, dropout=0.2, nhead=8):
        super().__init__()
        # MultiheadAttention expects (L, N, E) when batch_first=False
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=nhead, dropout=0.1, batch_first=False)
        self.dropout_attn = nn.Dropout(dropout)
        # Norm with BatchNorm1d around transpose as in config
        self.norm_attn = nn.Sequential(
            Transpose(1,2),
            nn.BatchNorm1d(embed_dim),
            Transpose(1,2)
        )
        # FFN
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        self.norm_ffn = nn.Sequential(
            Transpose(1,2),
            nn.BatchNorm1d(embed_dim),
            Transpose(1,2)
        )
    def forward(self, x):
        # x expected (seq, batch, embed)
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout_attn(attn_output)
        x = self.norm_attn(x)
        ffn_out = self.ff(x)
        x = x + self.dropout_ffn(ffn_out)
        x = self.norm_ffn(x)
        return x

class _TSTEncoder(nn.Module):
    def __init__(self, n_layers=4, embed_dim=128, ff_dim=256, dropout=0.2, nhead=8):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(_TSTEncoderLayer(embed_dim=embed_dim, ff_dim=ff_dim, dropout=dropout, nhead=nhead))
        self.layers = nn.ModuleList(layers)
    def forward(self, x):
        # x expected (seq, batch, embed)
        for layer in self.layers:
            x = layer(x)
        return x

class _TSTBackbone(nn.Module):
    def __init__(self, input_channels=4, embed_dim=128, kernel_size=25, dropout=0.2, n_layers=4):
        super().__init__()
        # W_P: Conv1d(4,128,kernel_size=25,padding=12)
        self.W_P = nn.Conv1d(in_channels=input_channels, out_channels=embed_dim, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout)
        self.encoder = _TSTEncoder(n_layers=n_layers, embed_dim=embed_dim, ff_dim=embed_dim*2, dropout=dropout)
        self.transpose = Transpose(-1, -2)
    def forward(self, x):
        # x: accept (batch, seq_len, channels) or (batch, channels, seq_len)
        if x.dim() == 3 and x.size(2) == self.W_P.in_channels:
            # convert to (batch, channels, seq)
            x = x.permute(0,2,1)
        # Apply conv: input should be (batch, channels, seq)
        x = self.W_P(x)  # (batch, embed_dim, seq)
        x = self.dropout(x)
        # Convert to (seq, batch, embed) for MultiheadAttention with batch_first=False
        x = x.permute(2, 0, 1)
        x = self.encoder(x)
        # After encoder, permute back to (batch, seq, embed)
        x = x.permute(1, 0, 2)
        return x

class TSTPlus(nn.Module):
    def __init__(self, input_channels=4, embed_dim=128, n_layers=4, seq_len=None, num_outputs=4, head_in_features=25728, nhead=8):
        super().__init__()
        self.backbone = _TSTBackbone(input_channels=input_channels, embed_dim=embed_dim, kernel_size=25, dropout=0.2, n_layers=n_layers)
        # head: GELU -> Flatten -> Dropout(0.1) -> Linear(head_in_features, num_outputs)
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(head_in_features, num_outputs)
        )
        self._meta = dict(embed_dim=embed_dim, n_layers=n_layers, head_in_features=head_in_features)
    def forward(self, x):
        out = self.backbone(x)
        out = self.head(out)
        return out

def get_model_by_name(name, **kwargs):
    """
    Registry helper: returns model instances by name.
    Note: If you want to use TSTPlus in your experiments, call get_model_by_name('tstplus', input_channels=4, head_in_features=25728, ...)
    """
    name = name.lower()
    if name == "tstplus":
        return TSTPlus(**kwargs)
    # fallback to searching globals() for model classes by name
    candidate = globals().get(name) or globals().get(name.upper())
    if isinstance(candidate, type):
        return candidate(**kwargs)
    raise ValueError(f"Model {name} not found. Available special model: TSTPlus")

# End of script
print("Plots will be saved to ./plots/ and tables to ./tables/ .")
