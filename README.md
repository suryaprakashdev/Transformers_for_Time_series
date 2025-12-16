# Transformer Models for Marker Top Prediction



## Description

GeoXts is a library for benchmarking time series classification. It is more focused on Geological usecase of trying to understand the sub-surface soil formations around the oil wells. 
The formation tops are refferred to as Markers. At present the marker depths are calculsted by expert geologist. This library is manly focussed on the automization of this process, for the new wells drilled.

We have an implementation of GradCAM focussed on time series data, which helps to understand the feature importance in multivariate time series. 


## Components of GeoXts

![Block diagram of the key components of GeoXts](/Images/Block_diagram.png) 


## Data files

The datasets of well logs and the marker depths have been extracted from the state government websites of USA. The datset for each well is extrated seperately and then joined together to create the whole dataset.
    
A sample dataset is provided, just to understant the format of the final created dataset. 

- [ ] [Colorado dataset](https://ecmc.state.co.us/data.html#/cogis) 
- [ ] [Wyoming dataset](https://pipeline.wyo.gov/wellchoiceMenu2.cfm?oops=ID88107&Skip=Y) 
- [ ] [UEA](https://www.timeseriesclassification.com/) is a benchmark dataset for multivariate time series classification. It can be downloaded from the website or using the tsai library.



[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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
   - Compact and efficient architecture
   - Optimized for computational performance

### Baseline Models (Previous Work)
- XCM
- LSTM
- Bi-LSTM

## 📊 Dataset

- **Type**: Univariate time-series well log data
- **Task**: Marker top prediction (classification)
- **Input Format**: Fixed-length time-series windows
- **Output Format**: Marker top class labels
- **Domain**: Geological/Geophysical data

## 🏆 Results

### Performance Comparison

| Model | Recall | Accuracy | Parameters | Notes |
|-------|--------|----------|------------|-------|
| **PatchTST** | **99%** | 98% | 796,676| **Best performance** on current dataset |
| **TSTPlus** | **99%** | 99.4% | 554,240 | More compact and efficient architecture |
| XCM (Baseline) | **96%** | 97.61% | 585,342 | Previous benchmark |
| LSTM (Baseline) | **96%** | 97.48% |415,918 | Previous benchmark |
| Bi-LSTM (Baseline) | **96%** | 96.66% | 148,654 | Previous benchmark |

### Key Findings
- ✅ PatchTST achieves **99% recall**, representing the best performance on the current dataset
- ✅ TSTPlus provides excellent performance with improved computational efficiency
- ✅ Transformer models demonstrate competitive or superior performance compared to traditional deep learning baselines





