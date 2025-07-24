# fuel-data-regression-model-with-0.42-mape
# High Accuracy Regression with LightGBM + CatBoost

This repository contains a high-accuracy regression pipeline that combines LightGBM and CatBoost into a robust ensemble for predicting multiple continuous targets.

> ✅ This version has been fully de-identified to preserve data confidentiality.

---

## 🔍 Overview

* 📈 Multi-target regression
* 🧠 Custom feature engineering: interaction terms, top-N correlation features, and aggregated impact scores
* 📊 5-fold cross-validation with out-of-fold prediction monitoring
* ⚖️ Weighted ensemble of LightGBM and CatBoost models (0.3 + 0.7)

## 📉 Accuracy

| Model        | Avg MAPE (Mean Absolute Percentage Error) |
| ------------ | ----------------------------------------- |
| LightGBM     | \~0.61                                    |
| CatBoost     | \~0.48                                    |
| **Ensemble** | **0.42**                                  |

> ⚠️ Note: These scores were obtained on private competition data and are shown for demonstration only. This repository does **not** contain real data.

---

## 🧪 How to Run

1. Clone this repository:

```bash
git clone https://github.com/your_username/high-accuracy-regression.git
cd high-accuracy-regression
```

2. Place your own `train.csv` and `test.csv` in the root directory.
3. Modify the `path_to_train.csv` and `path_to_test.csv` lines in `src/train.py`.
4. Run the model:

```bash
python src/train.py
```

---

## 🧠 Feature Engineering

The pipeline uses domain-agnostic strategies:

* Top 30 interaction features based on average correlation with targets
* Impact features calculated via weighted correlations
* Component-wise aggregation of feature importance

All identifiers have been anonymized (`Feature_F1`, `Feature_P3_4`, etc.), ensuring that no private feature names or real data values are exposed.

---

## 📂 Project Structure

```
.
├── src
│   └── train.py              # Main model pipeline (de-identified)
├── README.md
├── requirements.txt         # Python dependencies
└── .gitignore               # Ignore CSVs and outputs
```

---

## 📦 Requirements

```txt
pandas
numpy
scikit-learn
lightgbm
catboost
```

Install them via:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Disclaimer

This repo contains **no competition or proprietary data**. The model structure is shared strictly for educational and open-source demonstration purposes.

For questions or collaboration ideas, feel free to open an issue or drop a DM!

---

Made with 🔥 by \[YourName]
****
