# Optiver Realized Volatility Prediction

An end-to-end quantitative pipeline designed to forecast 10-minute realized stock volatility from tick-level order book and trade data across 112 equities. Built for the [Kaggle Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction) benchmark.

## Pipeline Architecture

- **High-Throughput Feature Extraction (`Polars`):** Streamlined feature transformation on partitioned Parquet files, computing multi-window microstructure signals:
  - **Weighted Average Price (WAP):** Derived 1st and 2nd tier WAP formulations to capture volume-weighted pricing pressure.
  - **Order Book Imbalance (OBI) & Spreads:** Bid-ask spread metrics and liquidity imbalance across order book depths.
  - **Multi-Window Realized Volatility:** Computed realized volatility across rolling sub-intervals (last 150s, 300s, and full 600s buckets).
- **Market-Wide Cross-Sectional Signals:** Synchronized stock-level metrics by `time_id` to generate broad market volatility mean, dispersion, and individual-to-market relative volatility ratios.
- **Leakage-Free Cross-Validation:** Partitioned validation splits using a 5-fold **`GroupKFold` grouped strictly by `time_id`**, guaranteeing zero cross-asset lookahead bias or temporal information leakage.
- **Custom Loss Formulation:** Trained an ensemble of LightGBM regressors with inverse-squared sample weighting ($w_i = 1 / y_i^2$) to align mean squared error directly with Root Mean Squared Percentage Error (RMSPE).

---

## Results & Performance

Evaluated against the competition's private test partition under the evaluation metric:

$$\text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$

| Model | CV RMSPE (OOF) | Private Test RMSPE | Improvement vs. Baseline |
| :--- | :---: | :---: | :---: |
| **Naive Historical Baseline** | — | 0.33750 | — |
| **LightGBM Ensemble (5-Fold GKF)** | **0.23151** | **0.23107** | **+31.5%** |

---

## Repository Structure

```text
├── src/
│   ├── features.py       # Polars feature extraction for book & trade data
│   └── train.py          # GroupKFold LightGBM training & RMSPE evaluation
├── notebooks/
│   └── pipeline.ipynb    # Kaggle-ready execution notebook
├── requirements.txt      # Environment dependencies
└── README.md
