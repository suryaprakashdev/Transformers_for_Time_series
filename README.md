
# Transformer Models for Marker Top Prediction
## 🎯 Overview
This project evaluates transformer-based architectures for predicting marker tops from time-series well log data. The work benchmarks modern transformer models against previously established deep learning baselines (XCM, LSTM, and Bi-LSTM) to assess whether transformers provide meaningful improvements in prediction quality and computational efficiency for this geological time-series task.
### Key Objectives
- Benchmark transformer architectures on well log time-series data
- Compare performance with existing deep learning baselines
- Evaluate prediction quality and model efficiency
- Identify optimal architectures for marker top prediction
## 🤖 Models
### Implemented Architectures
1. **PatchTST** (Patch-based Time Series Transformer)
   - Utilizes patch-based segmentation for efficient processing
   - Achieves state-of-the-art performance on the dataset

2. **TSTPlus** (Enhanced Time Series Transformer)
   - Compact and efficient architecture with Conv1D stem and learned positional embeddings
   - Optimized for computational performance

3. **TimesFM 2.5** (Google DeepMind Foundation Model)
   - 200M-parameter decoder-only transformer pre-trained on large-scale time-series corpora
   - Adapted via fine-tuning; only classification head trained (~800K trainable parameters)

4. **MANTIS** (Multi-scale Attention Network for Time-series Interpretation and Segmentation)
   - Hierarchical multi-scale attention operating at 1×, 2×, and 4× resolutions
   - Particularly suited to detecting both sharp and broad geological signatures

5. **MOMENT** (BERT-inspired Time-Series Transformer)
   - Uses a learnable CLS token for global sequence aggregation
   - Conv1D input embedding followed by a standard transformer encoder

### Baseline Models (Previous Work)
- XCM
- LSTM
- Bi-LSTM
- LSTM-2D CNN

## 📊 Dataset
- **Type**: Multivariate time-series well log data (Gamma Ray, Depth, Latitude, Longitude)
- **Task**: Geological marker top prediction (4-class classification: No Top, NIOBRARA, CODELL, FORT HAYS)
- **Input Format**: Fixed-length sliding windows of 201 samples
- **Output Format**: Marker top class labels
- **Domain**: Geological/Geophysical data — Colorado Well-Log Dataset (Denver-Julesburg Basin)

## 🏆 Results
### Performance Comparison
| Model | Accuracy | Recall | Precision | F1 | MAE (m) | Params (M) | Notes |
|-------|----------|--------|-----------|-----|---------|------------|-------|
| **TSTPlus** | **99.4%** | 0.981 | 0.993 | 0.987 | **1.00** | **0.5** | **Best overall** |
| **PatchTST** | **98.7%** | **0.990** | 0.988 | 0.989 | 1.89 | 1.2 | Best recall |
| MOMENT | 96.8% | 0.967 | 0.964 | 0.965 | 2.45 | 1.8 | Best non-patched transformer |
| TimesFM 2.5 | 96.0% | 0.955 | 0.951 | 0.953 | 3.21 | ≈200 | Frozen backbone |
| LSTM-XCM (Baseline) | 97.0% | 0.963 | 0.966 | 0.964 | 4.10 | 2.4 | Best baseline |
| LSTM-2D CNN (Baseline) | 96.1% | 0.958 | 0.956 | 0.957 | 4.72 | 2.1 | — |
| BiLSTM (Baseline) | 95.3% | 0.947 | 0.948 | 0.947 | 5.41 | 1.5 | — |
| LSTM (Baseline) | 94.8% | 0.941 | 0.942 | 0.941 | 5.83 | 0.8 | — |
| MANTIS | 94.2% | 0.941 | 0.938 | 0.939 | 4.12 | 3.8 | Best FORT HAYS recall (0.971) |

### Key Findings
- ✅ **TSTPlus** achieves the highest accuracy of **99.4%** with the fewest parameters (0.5M) and fastest supervised inference (1.1 ms/sample) — recommended for production deployment
- ✅ **PatchTST** achieves **99% recall** and 98.7% accuracy, outperforming all CNN/LSTM baselines
- ✅ **Patch-based tokenisation** is the key architectural ingredient — TSTPlus reduces depth MAE by **75%** (4.10 m → 1.00 m) over the best baseline
- ✅ **MOMENT** (96.8%) surpasses LSTM-XCM without patching, demonstrating the viability of CLS-token global aggregation
- ✅ **TimesFM** achieves competitive accuracy (96.0%) via transfer learning with a frozen backbone, useful when labelled data is scarce (<50 wells)
- ⚠️ **MANTIS** falls below the best baseline overall but excels on thick carbonate formations (FORT HAYS recall: 0.971); longer context windows (L=401) are recommended
- ⚠️ **TimesFM** inference latency (38.2 ms/sample) limits real-time applicability; supervised architectures are preferred for low-latency deployments
