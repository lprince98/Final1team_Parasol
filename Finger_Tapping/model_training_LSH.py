import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

import random
import click
import shap

from tqdm import tqdm
from pandas import DataFrame
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, cohen_kappa_score,
    r2_score, mean_absolute_percentage_error, accuracy_score,
    roc_auc_score
)
from imblearn.over_sampling import SMOTE

# =====================================
# SHAP 기반 Feature Selection
# =====================================
class BoostRFE:
    def __init__(self, base_model, n_features_to_select=20):
        self.base_model = base_model
        self.n_features_to_select = n_features_to_select
        self.support_ = None
        self.ranking_ = None
    
    def fit(self, X, y):
        features = list(X.columns)
        ranking = []

        while len(features) > self.n_features_to_select:
            model = self.base_model.fit(X[features], y)
            explainer = shap.TreeExplainer(model)
            shap_values = np.abs(explainer.shap_values(X[features])).mean(axis=0)

            worst_idx = np.argmin(shap_values)
            ranking.append(features[worst_idx])
            features.pop(worst_idx)

        self.support_ = features
        self.ranking_ = ranking[::-1] + features
        return self

class BoostRFA:
    def __init__(self, base_model, max_features=None):
        self.base_model = base_model
        self.max_features = max_features
        self.support_ = None
        self.ranking_ = None
    
    def fit(self, X, y):
        features = list(X.columns)
        model = self.base_model.fit(X[features], y)
        explainer = shap.TreeExplainer(model)
        shap_values = np.abs(explainer.shap_values(X[features])).mean(axis=0)

        sorted_idx = np.argsort(-shap_values)
        ordered_features = [features[i] for i in sorted_idx]

        if self.max_features is not None:
            self.support_ = ordered_features[:self.max_features]
        else:
            self.support_ = ordered_features

        self.ranking_ = ordered_features
        return self

class BoostBoruta:
    def __init__(self, base_model, n_iter=20):
        self.base_model = base_model
        self.n_iter = n_iter
        self.support_ = None
        self.ranking_ = None
    
    def fit(self, X, y):
        features = list(X.columns)
        confirmed_features = []

        for _ in range(self.n_iter):
            model = self.base_model.fit(X[features], y)
            explainer = shap.TreeExplainer(model)
            shap_values = np.abs(explainer.shap_values(X[features])).mean(axis=0)

            shadow = X.copy()
            for col in shadow.columns:
                shadow[col] = np.random.permutation(shadow[col].values)
            
            shadow_model = self.base_model.fit(shadow, y)
            shadow_explainer = shap.TreeExplainer(shadow_model)
            shadow_values = np.abs(shadow_explainer.shap_values(shadow)).mean(axis=0)

            for i, f in enumerate(features):
                if shap_values[i] > np.max(shadow_values):
                    confirmed_features.append(f)

        confirmed_features = list(set(confirmed_features))
        self.support_ = confirmed_features
        self.ranking_ = confirmed_features
        return self

# =====================================
# 데이터 로드
# =====================================
RATING_FILE = r"D:\workspace\Project\05_Final\code\finger\severity_dataset_dropped_correlated_columns_modf.csv"

def load():
    df = pd.read_csv(RATING_FILE)
    features = df.loc[:, 'wrist_mvmnt_x_median':'acceleration_min_trimmed']
    labels = df["diagnosed"]

    # 라벨이 문자열일 경우 숫자로 변환 (Series 유지)
    if labels.dtype == 'object':
        le = LabelEncoder()
        labels = pd.Series(le.fit_transform(labels), index=df.index)

    def parse(name: str):
        if name.startswith("NIH"):
            [ID, *_] = name.split("-")
        else:
            [*_, ID, _, _] = name.split("-")
        return ID
    
    df["id"] = df.filename.apply(parse)
    return features, labels, df["id"]

# =====================================
# Feature Selection
# =====================================
def select(features: DataFrame, labels, **cfg):
    methods = {"BoostRFE": BoostRFE, "BoostRFA": BoostRFA, "BoostBoruta": BoostBoruta}
    SELECTOR = methods[cfg["selector"]]

    base = XGBRegressor() if cfg["selector_base"] == "XGB" else LGBMRegressor()
    
    selector = SELECTOR(base)
    selector.fit(features, labels)

    selected = selector.support_
    features = features[selected]

    return features, labels

# =====================================
# Metrics
# =====================================
def metrics(preds, labels):
    results = {}

    preds, labels = np.array(preds), np.array(labels)
    results["mae"] = mean_absolute_error(labels, preds)
    results["mse"] = mean_squared_error(labels, preds)
    results["r2"] = r2_score(labels, preds)
    results["mape"] = mean_absolute_percentage_error(labels + 1, preds + 1)
    results["pearsonr"], _ = stats.pearsonr(labels, preds)

    rounded_preds = np.round(preds)
    rounded_labels = np.round(labels)
    results["kappa.no.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights=None)
    results["kappa.linear.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="linear")
    results["kappa.quadratic.weight"] = cohen_kappa_score(rounded_labels, rounded_preds, weights="quadratic")
    results["accuracy"] = accuracy_score(rounded_labels, rounded_preds)
    results["kendalltau"], _ = stats.kendalltau(labels, preds)
    results["spearmanr"], _ = stats.spearmanr(labels, preds)

    # AUC 추가 (이진 분류일 경우에만)
    try:
        if len(np.unique(labels)) == 2:
            results["auc"] = roc_auc_score(labels, preds)
    except Exception:
        results["auc"] = None

    return results

# =====================================
# 모델 정의
# =====================================
def model(**cfg):
    if cfg["model"] == "AdaBoostRegressor":
        return AdaBoostRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
        )
    if cfg["model"] == "RandomForestRegressor":
        return RandomForestRegressor(
            max_depth=cfg["max_depth"],
            max_features=cfg["max_features"],
            n_estimators=cfg["n_estimators"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=cfg["random_state"],
        )
    if cfg["model"] == "SVR":
        return SVR(C=cfg["C"], gamma=cfg["gamma"])
    if cfg["model"] == "LGBMRegressor":
        return LGBMRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            subsample=cfg["subsample"],
            random_state=cfg["random_state"],
        )
    if cfg["model"] == "XGBRegressor":
        return XGBRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"]
        )
    raise ValueError("Unknown model")

# =====================================
# Main Loop (Leave-One-Patient-Out CV)
# =====================================
@click.command()
@click.option("--model", default="LGBMRegressor", help="Model to use")
@click.option("--selector", default="BoostRFE", help="Feature selection method")
@click.option("--selector_base", default="LGBM", help="Base regressor for feature selection")
@click.option("--n_estimators", default=611, help="Number of estimators for regressor")
@click.option("--learning_rate", default=0.01313, help="Learning rate for regressor")
@click.option("--max_depth", default=3, help="Max depth for regressor")
@click.option("--colsample_bytree", default=0.8, help="Colsample by tree for regressor")
@click.option("--subsample", default=0.8, help="Subsample for regressor")
@click.option("--reg_alpha", default=0.1, help="Reg alpha for regressor")
@click.option("--reg_lambda", default=0.1, help="Reg lambda for regressor")
@click.option("--max_features", default="sqrt", help="Max features for regressor")
@click.option("--min_samples_split", default=2, help="Min samples split for regressor")
@click.option("--min_samples_leaf", default=1, help="Min samples leaf for regressor")
@click.option("--min_child_weight", default=1, help="Min child weight for regressor")
@click.option("--C", default=1.0, help="C for regressor")
@click.option("--gamma", default="scale", help="Gamma for regressor")
@click.option("--n", default=22, help="Number of features to select")
@click.option("--random_state", default=42, help="Random state for regressor")
@click.option("--seed", default=42, help="Seed for random")
@click.option("--use_feature_selection", default='yes', help="yes/no")
@click.option("--use_feature_scaling", default='yes', help="yes/no")
@click.option("--scaling_method", default='StandardScaler', help="StandardScaler / MinMaxScaler")
@click.option("--minority_oversample", default='no', help="yes/no")
def main(**cfg):
    features, labels, ids = load()

    if cfg["use_feature_selection"] == 'yes':
        features, labels = select(features, labels, **cfg)
    
    regressor = model(**cfg)

    all_preds = []
    all_labels = []
    save_preds = pd.read_csv(RATING_FILE)
    save_preds["preds"] = save_preds["diagnosed"]

    loo = LeaveOneGroupOut()
    oversample = SMOTE(random_state=cfg['random_state'])

    for train_index, test_index in tqdm(loo.split(features, groups=ids), total=len(ids.unique())):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        if cfg['use_feature_scaling'] == 'yes':
            if cfg['scaling_method'] == 'StandardScaler':
                scaler = StandardScaler()
            else:
                scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if cfg['minority_oversample'] == 'yes':
            (X_train, y_train) = oversample.fit_resample(X_train, y_train)

        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)

        all_preds.extend(y_pred)
        all_labels.extend(y_test)
        save_preds.loc[test_index, "preds"] = y_pred

    results = metrics(all_preds, all_labels)
    print("Final result:", results)

if __name__ == "__main__":
    main()
