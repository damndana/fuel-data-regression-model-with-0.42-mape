#High Accuracy Regression Pipeline (De-Identified)
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import lightgbm as lgb
import catboost as cb

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

fraction_cols = [col for col in train_df.columns if "Feature_F" in col]
property_cols = [col for col in train_df.columns if "Feature_P" in col]
target_cols = [col for col in train_df.columns if "Target_" in col]

interaction_cols = []
for f_col in fraction_cols:
    for p_col in property_cols:
        inter_name = f"{f_col}_x_{p_col.split('_')[-1]}"
        train_df[inter_name] = train_df[f_col] * train_df[p_col]
        test_df[inter_name] = test_df[f_col] * test_df[p_col]
        interaction_cols.append(inter_name)

corr_matrix = train_df[interaction_cols + target_cols].corr()
avg_corr = corr_matrix.abs().loc[interaction_cols, target_cols].mean(axis=1)
kept_interactions = avg_corr.sort_values(ascending=False).head(30).index.tolist()

all_features = fraction_cols + property_cols + kept_interactions
for target in target_cols:
    corr_sorted = train_df[all_features + [target]].corr()[target].abs().sort_values(ascending=False)
    top_features = corr_sorted.head(10).index.tolist()
    impact_name = f"Impact_{target}"
    train_df[impact_name] = sum([train_df[f] * train_df[[f, target]].corr().iloc[0, 1] for f in top_features if f in train_df])
    test_df[impact_name] = sum([test_df[f] * train_df[[f, target]].corr().iloc[0, 1] for f in top_features if f in test_df])

for i in range(1, 11):
    name = f"Top5_Impact_Target{i}"
    train_df[name] = 0
    test_df[name] = 0
    for j in range(1, 6):
        prop_col = f"Feature_P{j}_{i}"
        frac_col = f"Feature_F{j}"
        if prop_col in train_df and frac_col in train_df:
            train_df[name] += train_df[prop_col] * train_df[frac_col]
            test_df[name] += test_df[prop_col] * test_df[frac_col]

important_targets = [1, 2, 4, 6, 8, 9]
train_df['FeaturePowerImpact'] = 0
test_df['FeaturePowerImpact'] = 0
for i in important_targets:
    col = f"F5_x_P{i}"
    train_df[col] = train_df.get("Feature_F5", 0) * train_df.get(f"Feature_P5_{i}", 0)
    test_df[col] = test_df.get("Feature_F5", 0) * test_df.get(f"Feature_P5_{i}", 0)
    train_df['FeaturePowerImpact'] += train_df[col]
    test_df['FeaturePowerImpact'] += test_df[col]

feature_cols = [col for col in train_df.columns if col not in ['ID'] + target_cols]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lgb_test_preds, cb_test_preds = pd.DataFrame(), pd.DataFrame()
oof_lgb, oof_cb, oof_ensemble = pd.DataFrame(index=train_df.index), pd.DataFrame(index=train_df.index), pd.DataFrame(index=train_df.index)
mape_lgb, mape_cb, mape_ensemble = {}, {}, {}

train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)
if 'ID' not in test_df.columns:
    test_df['ID'] = range(1, len(test_df) + 1)

for col in feature_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)

for target in target_cols:
    oof_lgb_preds = np.zeros(len(train_df))
    oof_cb_preds = np.zeros(len(train_df))
    test_lgb_preds = np.zeros(len(test_df))
    test_cb_preds = np.zeros(len(test_df))

    for train_idx, val_idx in kf.split(train_df):
        X_train, y_train = train_df.iloc[train_idx][feature_cols], train_df.iloc[train_idx][target]
        X_val, y_val = train_df.iloc[val_idx][feature_cols], train_df.iloc[val_idx][target]

        lgb_model = lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.05, objective='huber', n_jobs=-1)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='l1', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_lgb_preds[val_idx] = lgb_model.predict(X_val)
        test_lgb_preds += lgb_model.predict(test_df[feature_cols]) / kf.n_splits

        cb_model = cb.CatBoostRegressor(random_state=42, iterations=1000, learning_rate=0.05, verbose=0, loss_function='MAE', early_stopping_rounds=100)
        cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        oof_cb_preds[val_idx] = cb_model.predict(X_val)
        test_cb_preds += cb_model.predict(test_df[feature_cols]) / kf.n_splits

    lgb_test_preds[target] = test_lgb_preds
    cb_test_preds[target] = test_cb_preds
    oof_lgb[target], oof_cb[target] = oof_lgb_preds, oof_cb_preds
    oof_ensemble[target] = 0.3 * oof_lgb_preds + 0.7 * oof_cb_preds
    mape_lgb[target] = mean_absolute_percentage_error(train_df[target], oof_lgb_preds)
    mape_cb[target] = mean_absolute_percentage_error(train_df[target], oof_cb_preds)
    mape_ensemble[target] = mean_absolute_percentage_error(train_df[target], oof_ensemble[target])

print("\nMAPE (LightGBM):")
for target in target_cols:
    print(f"{target}: {mape_lgb[target]:.4f}")

print("\nMAPE (CatBoost):")
for target in target_cols:
    print(f"{target}: {mape_cb[target]:.4f}")

print("\nMAPE (Ensemble):")
for target in target_cols:
    print(f"{target}: {mape_ensemble[target]:.4f}")

print("\nAverage MAPE (LightGBM):", np.mean(list(mape_lgb.values())))
print("Average MAPE (CatBoost):", np.mean(list(mape_cb.values())))
print("Average MAPE (Ensemble):", np.mean(list(mape_ensemble.values())))

ensemble_test_preds = 0.3 * lgb_test_preds + 0.7 * cb_test_preds
submission = pd.DataFrame({"id": test_df["ID"]})
for target in target_cols:
    submission[target] = ensemble_test_preds[target]
submission.to_csv("sample_submission_with_id.csv", index=False)

oof_ensemble.to_csv("oof_preds.csv", index=False)
ensemble_test_preds.to_csv("test_preds.csv", index=False)
train_df[target_cols].to_csv("y_true.csv", index=False)
print("Predictions and truth saved.")
