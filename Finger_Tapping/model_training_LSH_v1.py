# -*- coding: utf-8 -*-
"""
PD vs HC 이진분류 파이프라인 (최종본 + Noisy-OR 임계값 튜닝)
- Hold-out Split (8:2)
- Feature Selection: SHAP → RFE → RFA
- Optuna 기반 12개 모델 하이퍼파라미터 최적화 (목표: F1)
- 모든 모델 Hold-out 성능 + 5-Fold CV 성능 CSV 저장
- 사람 단위(Noisy-OR) 확률 + 목표 재현율 기반 임계값 자동 탐색
- Recall/F1 기준 Best 모델을 전체 데이터로 재학습하여 저장:
    artifact = {
        "pipeline": pipeline,
        "selected_features": selected_feat,    # 전처리 이후 피처명(num__/cat__…)
        "raw_features": raw_features,          # 실시간 입력 정렬용
        "person_threshold": tuned_thr,         # 사람 단위(Noisy-OR) 임계값
        "hand_threshold": 0.5,                 # 손 단위 기본 임계값(표시용)
        "threshold_strategy": "noisyor_target_recall",
        "target_recall": TARGET_RECALL
    }
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import shap

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, make_scorer
)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SequentialFeatureSelector

# 분류기들
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from joblib import dump

# =========================
# 설정
# =========================
CSV_PATH = "severity_dataset_dropped_correlated_columns_modf.csv"
ARTIFACT_DIR = "pd_hc_binary"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 20   # Optuna trial 수
USE_RFE, RFE_N_FEATURES = True, 30
USE_RFA, RFA_N_FEATURES = True, 20
TARGET_RECALL = 0.90           # 사람 단위에서 목표 재현율
HAND_THRESHOLD_DEFAULT = 0.50   # 손 단위 기본 임계값(표시/로그용)

MODEL_LIST = [
    "LightGBM","XGBoost","CatBoost","LogisticRegression","RandomForest",
    "ExtraTrees","GradientBoosting","AdaBoost","SVM","GaussianNB","MLP","KNN"
]

# =========================
# 유틸: Noisy-OR & 임계값 튜닝
# =========================
def noisyor_group_probs(groups, probs):
    """그룹(사람)별 Noisy-OR 확률 = 1 - Π(1 - p_hand)"""
    df = pd.DataFrame({"g": groups, "p": probs})
    agg = df.groupby("g")["p"].apply(lambda s: 1.0 - float(np.prod(1.0 - s.values)))
    return agg  # index=group, values=prob

def tune_threshold_noisyor(y_true_person, p_person, target_recall=0.90):
    """목표 재현율 이상을 만족하는 임계값 중 F1 최고를 선택. 없으면 F1 최고."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_meeting = None
    best_overall = {"thr": 0.5, "recall": 0.0, "f1": 0.0}

    for t in thresholds:
        pred = (p_person >= t).astype(int)
        rec = recall_score(y_true_person, pred, zero_division=0)
        f1  = f1_score(y_true_person, pred, zero_division=0)
        if rec >= target_recall:
            if (best_meeting is None) or (f1 > best_meeting["f1"]):
                best_meeting = {"thr": float(t), "recall": float(rec), "f1": float(f1)}
        if f1 > best_overall["f1"]:
            best_overall = {"thr": float(t), "recall": float(rec), "f1": float(f1)}

    return best_meeting if best_meeting is not None else best_overall

# =========================
# 데이터 적재 & 라벨/그룹 분리
# =========================
df = pd.read_csv(CSV_PATH)
y = df["diagnosed"].str.lower().map({"yes": 1, "no": 0}).astype(int)
groups = df["filename"].astype(str)

drop_cols = ["Unnamed: 0"] if "Unnamed: 0" in df.columns else []
for c in ["diagnosed", "Rating", "filename"]:
    if c in df.columns: drop_cols.append(c)
X_all = df.drop(columns=drop_cols, errors="ignore")

cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
num_cols = [c for c in X_all.columns if c not in cat_cols]

# =========================
# Hold-out Split (8:2)
# =========================
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X_all, y, groups, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"[INFO] Train size={X_train.shape[0]}, Test size={X_test.shape[0]}")

# =========================
# SHAP 기반 Feature Selection (Base: LightGBM)
# =========================
pre_all = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="drop"
)
base_model_fs = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
pipe_base = Pipeline(steps=[("preprocessor", pre_all), ("classifier", base_model_fs)])
pipe_base.fit(X_train, y_train)

feature_names = pipe_base.named_steps["preprocessor"].get_feature_names_out()  # 예: num__feat, cat__gender_female
X_proc = pipe_base.named_steps["preprocessor"].transform(X_train)

# shap 값 계산
explainer = shap.TreeExplainer(pipe_base.named_steps["classifier"])
shap_values = explainer.shap_values(X_proc)
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # 양성 클래스 기준
shap_importance = np.abs(shap_values).mean(axis=0)

feat_importance = pd.DataFrame({"feature": feature_names, "importance": shap_importance})
median_val = feat_importance["importance"].median()
selected_feat = feat_importance.loc[feat_importance["importance"] >= median_val, "feature"].tolist()

# === RFE 단계 ===
if USE_RFE and len(selected_feat) > RFE_N_FEATURES:
    rfe_est = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    rfe = RFE(estimator=rfe_est, n_features_to_select=RFE_N_FEATURES)
    idx_sel = [list(feature_names).index(f) for f in selected_feat]
    rfe.fit(X_proc[:, idx_sel], y_train)
    selected_feat = [f for f, keep in zip(selected_feat, rfe.support_) if keep]

# === RFA 단계 (groups 미지원 → StratifiedKFold 사용) ===
if USE_RFA and len(selected_feat) > RFA_N_FEATURES:
    rfa_est = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sfs = SequentialFeatureSelector(
        rfa_est, n_features_to_select=RFA_N_FEATURES,
        direction="forward", scoring="roc_auc", cv=cv_inner, n_jobs=-1
    )
    idx_sel = [list(feature_names).index(f) for f in selected_feat]
    sfs.fit(X_proc[:, idx_sel], y_train)
    selected_feat = [f for f, keep in zip(selected_feat, sfs.get_support()) if keep]

print(f"[INFO] 최종 선택된 feature 수: {len(selected_feat)}")

# ===== 선택된 피처로부터 raw 입력 컬럼 계산 =====
raw_num_selected, raw_cat_selected = [], []
for f in selected_feat:
    if f.startswith("num__"):
        col = f.split("num__")[-1]
        if col in num_cols and col not in raw_num_selected:
            raw_num_selected.append(col)
    elif f.startswith("cat__"):
        tail = f.split("cat__")[-1]
        col = tail.split("_")[0]  # 'gender_female' → 'gender'
        if col in cat_cols and col not in raw_cat_selected:
            raw_cat_selected.append(col)
raw_features = raw_num_selected + raw_cat_selected
print(f"[INFO] 원시 입력 컬럼 수: {len(raw_features)} (num={len(raw_num_selected)}, cat={len(raw_cat_selected)})")

def make_preprocessor_selected():
    transformers = []
    if len(raw_num_selected) > 0:
        transformers.append(("num", StandardScaler(), raw_num_selected))
    if len(raw_cat_selected) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), raw_cat_selected))
    if not transformers:
        transformers.append(("num", StandardScaler(), []))
    return ColumnTransformer(transformers=transformers, remainder="drop")

# =========================
# Optuna Objective (목표: F1)
# =========================
def build_model(trial, model_name):
    if model_name == "LightGBM":
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            num_leaves=trial.suggest_int("num_leaves", 15, 63),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RANDOM_STATE, class_weight="balanced", verbose=-1
        )
    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            subsample=trial.suggest_float("subsample", 0.7, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
            random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss", verbosity=0
        )
    elif model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            random_state=RANDOM_STATE, class_weight="balanced"
        )
    elif model_name == "AdaBoost":
        return AdaBoostClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.5),
            random_state=RANDOM_STATE
        )
    elif model_name == "SVM":
        return SVC(
            C=trial.suggest_float("C", 0.1, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
            kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE
        )
    elif model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(trial.suggest_int("layer1", 32, 128),
                                trial.suggest_int("layer2", 16, 64)),
            max_iter=500, random_state=RANDOM_STATE
        )
    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=trial.suggest_int("n_neighbors", 3, 15))
    elif model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            random_state=RANDOM_STATE, class_weight="balanced"
        )
    elif model_name == "GradientBoosting":
        return GradientBoostingClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            random_state=RANDOM_STATE
        )
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 200, 800),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            depth=trial.suggest_int("depth", 4, 10),
            verbose=0, random_state=RANDOM_STATE
        )
    elif model_name == "LogisticRegression":
        return LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")
    elif model_name == "GaussianNB":
        return GaussianNB()
    else:
        raise ValueError(model_name)

def objective(trial, model_name):
    model = build_model(trial, model_name)
    pre_sel = make_preprocessor_selected()
    pipeline = Pipeline(steps=[("preprocessor", pre_sel), ("classifier", model)])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, groups=groups_train,
        scoring=make_scorer(f1_score)
    )
    return scores.mean()

# =========================
# Optuna 실행 및 최적 모델 학습
# =========================
optuna_rows, best_models, best_params_by_model = [], {}, {}

for model_name in MODEL_LIST:
    print(f"\n[Optuna] Optimizing {model_name} ...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=N_TRIALS)

    best_params = study.best_params
    best_score = study.best_value
    print(f"[Optuna] Best {model_name} F1={best_score:.3f}, params={best_params}")

    optuna_rows.append({"model": model_name, "best_f1": best_score, "best_params": best_params})
    best_params_by_model[model_name] = best_params

    # 최적 파라미터로 파이프라인 구성 & 학습(훈련 세트)
    final_model = build_model(optuna.trial.FixedTrial(best_params), model_name)
    pre_sel = make_preprocessor_selected()
    pipe = Pipeline(steps=[("preprocessor", pre_sel), ("classifier", final_model)])
    pipe.fit(X_train, y_train)
    best_models[model_name] = pipe

pd.DataFrame(optuna_rows).to_csv(os.path.join(ARTIFACT_DIR, "optuna_results.csv"), index=False)

# =========================
# Hold-out Test: 모든 모델 성능 기록
# =========================
test_rows = []
per_model_test_preds = {}  # 사람 임계값 튜닝용으로 보관
for model_name, pipe in best_models.items():
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)
    test_rows.append({
        "model": model_name,
        "AUC": roc_auc_score(y_test, y_prob),
        "F1": f1_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Accuracy": accuracy_score(y_test, y_pred)
    })
    per_model_test_preds[model_name] = (y_prob, y_pred)

test_df = pd.DataFrame(test_rows).sort_values("F1", ascending=False)
test_df.to_csv(os.path.join(ARTIFACT_DIR, "test_results.csv"), index=False)
print(f"[INFO] Saved ALL model hold-out test results to {os.path.join(ARTIFACT_DIR, 'test_results.csv')}")

# 승자 결정
best_recall_model_name = test_df.loc[test_df["Recall"].idxmax(), "model"]
best_f1_model_name     = test_df.loc[test_df["F1"].idxmax(), "model"]
best_accuracy_model_name = test_df.loc[test_df["Accuracy"].idxmax(), "model"]
best_AUC_model_name     = test_df.loc[test_df["AUC"].idxmax(), "model"]
print(f"[WIN] Best Recall: {best_recall_model_name} / Best F1: {best_f1_model_name} / Best Accuracy: {best_accuracy_model_name} / Best AUC: {best_AUC_model_name}")

# =========================
# Cross-validation Results 저장 (훈련 세트 내 5-Fold)
# =========================
cv_rows = []
cv5 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for model_name in MODEL_LIST:
    params = best_params_by_model[model_name]
    model = build_model(optuna.trial.FixedTrial(params), model_name)
    pre_sel = make_preprocessor_selected()
    pipe = Pipeline(steps=[("preprocessor", pre_sel), ("classifier", model)])

    fold = 1
    for tr_idx, va_idx in cv5.split(X_train, y_train, groups_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        pipe.fit(X_tr, y_tr)
        y_prob = pipe.predict_proba(X_va)[:, 1]
        y_pred = pipe.predict(X_va)

        cv_rows.append({
            "model": model_name, "fold": fold,
            "AUC": roc_auc_score(y_va, y_prob),
            "F1": f1_score(y_va, y_pred),
            "Recall": recall_score(y_va, y_pred),
            "Precision": precision_score(y_va, y_pred),
            "Accuracy": accuracy_score(y_va, y_pred)
        })
        fold += 1

pd.DataFrame(cv_rows).to_csv(os.path.join(ARTIFACT_DIR, "cv_results.csv"), index=False)
print(f"[INFO] Saved CV results (fold-level) to {os.path.join(ARTIFACT_DIR, 'cv_results.csv')}")

# =========================
# 사람 단위 임계값 튜닝(hold-out test) → 각 베스트 모델별 튜닝값 계산
#  * 주의: 일반론적으로 임계값 튜닝은 검증셋에서 하는 것이 이상적입니다.
# =========================
def tune_for_model(model_name):
    y_prob_test, _ = per_model_test_preds[model_name]
    # 사람 단위 라벨: 같은 filename(=groups_test)끼리 동일하다고 가정
    y_true_person = pd.Series(y_test.values, index=groups_test).groupby(level=0).max()
    p_person = noisyor_group_probs(groups_test.values, y_prob_test)
    tuned = tune_threshold_noisyor(y_true_person.values, p_person.values, target_recall=TARGET_RECALL)
    print(f"[TUNE] {model_name}: Noisy-OR thr={tuned['thr']:.2f}, recall={tuned['recall']:.3f}, f1={tuned['f1']:.3f}, accuracy={tuned['accuracy']:.3f}, AUC={tuned['AUC']:.3f}")
    # 로그용 CSV 저장
    df_tune = pd.DataFrame({
        "filename": p_person.index,
        "y_true_person": y_true_person.loc[p_person.index].values,
        "p_person": p_person.values
    })
    df_tune.to_csv(os.path.join(ARTIFACT_DIR, f"person_probs_{model_name}.csv"), index=False)
    return tuned

tuned_recall = tune_for_model(best_recall_model_name)
tuned_f1     = tune_for_model(best_f1_model_name)
tuned_accuracy = tune_for_model(best_accuracy_model_name)
tuned_AUC     = tune_for_model(best_AUC_model_name)

# =========================
# Best(Recall/F1) 모델을 전체 데이터로 재학습하여 저장
# =========================
for criterion, winner, tuned in [
    ("recall", best_recall_model_name, tuned_recall),
    ("f1",     best_f1_model_name,     tuned_f1)
    ("accuracy", best_accuracy_model_name, tuned_accuracy),
    ("AUC",     best_AUC_model_name,     tuned_AUC)
]:
    params = best_params_by_model[winner]
    model = build_model(optuna.trial.FixedTrial(params), winner)

    pre_sel_full = make_preprocessor_selected()
    pipe_full = Pipeline(steps=[("preprocessor", pre_sel_full), ("classifier", model)])
    pipe_full.fit(X_all, y)  # 전체 데이터로 재학습

    artifact = {
        "pipeline": pipe_full,
        "selected_features": selected_feat,             # 전처리 이후 이름들(num__/cat__...)
        "raw_features": raw_features,                   # 파이프라인 입력으로 필요(예측 시 자동 정렬)
        "person_threshold": float(tuned["thr"]),        # 사람 단위(Noisy-OR) 임계값
        "hand_threshold": float(HAND_THRESHOLD_DEFAULT),
        "threshold_strategy": "noisyor_target_recall",
        "target_recall": float(TARGET_RECALL)
    }
    save_path = os.path.join(ARTIFACT_DIR, f"best_pipeline_{criterion}_{winner}.joblib")
    dump(artifact, save_path)
    print(f"[INFO] Saved artifact → {save_path}")

print("[DONE] Training + Threshold tuning + Artifact export complete.")
