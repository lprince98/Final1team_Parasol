# -*- coding: utf-8 -*-
"""
PD vs HC 이진분류 파이프라인 (AUC 최고 모델 선정 및 종합 아티팩트 저장)
"""

# ==============================================================================
# 0. 라이브러리 임포트
# ==============================================================================
import os, warnings
warnings.filterwarnings("ignore") # 불필요한 경고 메시지 무시

# 데이터 처리 및 연산
import numpy as np
import pandas as pd

# 하이퍼파라미터 튜닝
import optuna

# 모델 해석 및 피처 중요도
import shap

# Scikit-learn: 모델링 및 평가 관련 도구
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score, accuracy_score, make_scorer, confusion_matrix
)

# 분류기 모델들
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# 최종 산출물 저장
from joblib import dump

# ==============================================================================
# 1. 전역 설정 및 상수 정의
# ==============================================================================
# Optuna 최적화 목표 지표 설정 ('auc' 또는 'f1')
OBJECTIVE_METRIC = 'auc'
# Scikit-learn의 cross_val_score에서 사용할 평가 지표 문자열
OBJECTIVE_SCORER = 'roc_auc' if OBJECTIVE_METRIC == 'auc' else make_scorer(f1_score)

# 데이터 및 결과물 경로 설정
CSV_PATH = "severity_dataset_dropped_correlated_columns_modf.csv" # 입력 데이터 파일 경로
ARTIFACT_DIR = "pd_hc_binary" # 결과물이 저장될 디렉토리
os.makedirs(ARTIFACT_DIR, exist_ok=True) # 디렉토리가 없으면 생성

# 실험 재현성을 위한 설정
RANDOM_STATE = 42 # 모든 랜덤 연산에 사용할 시드 값
N_TRIALS = 20 # Optuna가 각 모델별로 시도할 하이퍼파라미터 조합의 수

# 피처 선택 전략 설정
USE_RFE, RFE_N_FEATURES = True, 30 # RFE(재귀적 피처 제거) 사용 여부 및 목표 피처 수
USE_RFA, RFA_N_FEATURES = True, 20 # SFS(순차적 피처 선택, 코드에서는 RFA로 명명) 사용 여부 및 목표 피처 수

# 임계값 튜닝 관련 설정
TARGET_RECALL = 0.90 # 재현율 목표치 임계값 튜닝 시 사용할 타겟 재현율
HAND_THRESHOLD_DEFAULT = 0.50 # 손(hand) 단위 예측 시 사용할 기본 임계값 (현재 코드에서는 직접 사용되지 않음)

# 튜닝 및 평가를 진행할 모델 목록
MODEL_LIST = [
    "LightGBM", "XGBoost", "CatBoost", "LogisticRegression", "RandomForest",
    "ExtraTrees", "GradientBoosting", "AdaBoost", "SVM", "GaussianNB", "MLP", "KNN"
]

# ==============================================================================
# 2. 유틸리티 함수 정의 (주로 임계값 튜닝 관련)
# ==============================================================================

def noisyor_group_probs(groups, probs):
    """
    동일한 사람(group)에게서 나온 여러 예측 확률(probs)을 'noisy-OR' 방식으로 결합하여
    사람 단위의 단일 예측 확률을 계산합니다.
    (예: 한 사람의 왼손, 오른손 데이터 예측 결과를 합산)
    """
    df = pd.DataFrame({"g": groups, "p": probs})
    # 한 사람이라도 'yes'일 확률 = 1 - (모든 예측이 'no'일 확률의 곱)
    agg = df.groupby("g")["p"].apply(lambda s: 1.0 - float(np.prod(1.0 - s.values)))
    return agg

def tune_threshold_noisyor(y_true_person, p_person, target_recall=0.90):
    """
    사람 단위 예측에서, 목표 재현율(target_recall)을 만족시키면서
    F1 점수가 가장 높은 최적의 임계값(threshold)을 찾습니다.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_meeting = None # 재현율 목표를 만족하는 후보 중 최고
    best_overall = {"thr": 0.5, "recall": 0.0, "f1": 0.0} # 전체 후보 중 F1 최고

    for t in thresholds:
        pred = (p_person >= t).astype(int)
        rec = recall_score(y_true_person, pred, zero_division=0)
        f1  = f1_score(y_true_person, pred, zero_division=0)

        # 재현율 목표를 달성한 경우
        if rec >= target_recall:
            if (best_meeting is None) or (f1 > best_meeting["f1"]):
                best_meeting = {"threshold": float(t), "recall": float(rec), "f1": float(f1)}
        
        # 전체 F1 최고 기록 갱신
        if f1 > best_overall["f1"]:
            best_overall = {"threshold": float(t), "recall": float(rec), "f1": float(f1)}
    
    # 목표를 달성한 임계값이 있으면 그것을, 없으면 F1이 가장 높았던 임계값을 선택
    final_choice = best_meeting if best_meeting is not None else best_overall
    if "threshold" not in final_choice: final_choice["threshold"] = 0.5 # 예외 처리
    return final_choice

def tune_threshold_youden(y_true_person, p_person):
    """
    Youden's J-index (민감도 + 특이도 - 1)를 최대화하는 최적의 임계값을 찾습니다.
    이 지표는 모델이 양성과 음성을 얼마나 잘 구분하는지 종합적으로 나타냅니다.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best = {"threshold": 0.5, "sensitivity": 0.0, "specificity": 0.0, "youden": -1.0}
    for t in thresholds:
        pred = (p_person >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_person, pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # 재현율 (Recall)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        youden = sensitivity + specificity - 1
        if youden > best["youden"]:
            best = {
                "threshold": float(t), "sensitivity": float(sensitivity),
                "specificity": float(specificity), "youden": float(youden)
            }
    return best

def tune_threshold_by_metric(y_true, y_prob, metric_func, metric_name):
    """
    F1, Accuracy 등 특정 평가 지표(metric)를 최대화하는 범용 임계값 튜닝 함수입니다.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_score = -1
    best_thr = 0.5
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        
        # 평가 지표에 따라 인자 사용을 다르게 함
        if metric_name == "accuracy":
            score = metric_func(y_true, y_pred)
        else:
            score = metric_func(y_true, y_pred, zero_division=0)
            
        if score > best_score:
            best_score = score
            best_thr = thr
            
    return {"threshold": float(best_thr), metric_name: float(best_score)}

# ==============================================================================
# 3. 데이터 로딩 및 전처리, 분할
# ==============================================================================
# CSV 파일 로드
df = pd.read_csv(CSV_PATH)

# 타겟 변수(y) 생성: 'yes' -> 1, 'no' -> 0
y = df["diagnosed"].str.lower().map({"yes": 1, "no": 0}).astype(int)
# 그룹 변수(groups) 생성: 데이터가 어떤 사람/파일에서 왔는지 식별
groups = df["filename"].astype(str)

# 불필요한 컬럼들(인덱스, 타겟, 그룹 정보 등)을 피처에서 제외
drop_cols = ["Unnamed: 0"] if "Unnamed: 0" in df.columns else []
for c in ["diagnosed", "Rating", "filename"]:
    if c in df.columns: drop_cols.append(c)
X_all = df.drop(columns=drop_cols, errors="ignore")

# 피처를 범주형(categorical)과 수치형(numerical)으로 분리
cat_cols = [c for c in X_all.columns if X_all[c].dtype == "object"]
num_cols = [c for c in X_all.columns if c not in cat_cols]

# Hold-out: 전체 데이터를 학습(Train)용과 테스트(Test)용으로 분할 (80:20)
# stratify=y : 원본 데이터의 0/1 비율을 학습/테스트 데이터에서도 동일하게 유지
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X_all, y, groups, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"[INFO] Train size={X_train.shape[0]}, Test size={X_test.shape[0]}")

# ==============================================================================
# 4. SHAP 기반 피처 선택 (Feature Selection)
# ==============================================================================
# 1. SHAP 중요도 계산을 위한 기본 모델 파이프라인 생성
#    - 수치형 피처: StandardScaler로 스케일링
#    - 범주형 피처: OneHotEncoder로 변환
pre_all = ColumnTransformer(transformers=[("num", StandardScaler(), num_cols), 
                                          ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), 
                                           cat_cols)], remainder="drop")
base_model_fs = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
pipe_base = Pipeline(steps=[("preprocessor", pre_all), ("classifier", base_model_fs)])
pipe_base.fit(X_train, y_train)

# 2. SHAP 값 계산
feature_names = pipe_base.named_steps["preprocessor"].get_feature_names_out() # 전처리 후 피처 이름들
X_proc = pipe_base.named_steps["preprocessor"].transform(X_train) # 학습 데이터 전처리
explainer = shap.TreeExplainer(pipe_base.named_steps["classifier"]) # SHAP 설명객체 생성
shap_values = explainer.shap_values(X_proc) # SHAP 값 계산
if isinstance(shap_values, list): shap_values = shap_values[1] # 이진 분류의 경우 양성 클래스(1)에 대한 SHAP 값만 사용
shap_importance = np.abs(shap_values).mean(axis=0) # 각 피처의 평균적인 영향력(절대값 기준) 계산

# 3. SHAP 중요도 기반 1차 피처 선택
feat_importance = pd.DataFrame({"feature": feature_names, "importance": shap_importance})
feat_importance = feat_importance.sort_values('importance', ascending=False).reset_index(drop=True)
median_val = feat_importance['importance'].median() # 중요도의 중앙값 계산
feat_importance['selected_by_shap'] = feat_importance['importance'] >= median_val # 중앙값 이상인 피처만 선택
shap_csv_path = os.path.join(ARTIFACT_DIR, 'shap_features.csv')
feat_importance.to_csv(shap_csv_path, index=False)
print(f"[INFO] Saved SHAP features → {shap_csv_path}")

selected_feat = feat_importance.loc[feat_importance['selected_by_shap'], 'feature'].tolist()

# 4. (선택적) RFE를 이용한 2차 피처 선택
if USE_RFE and len(selected_feat) > RFE_N_FEATURES:
    rfe_est = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    rfe = RFE(estimator=rfe_est, n_features_to_select=RFE_N_FEATURES)
    idx_sel = [list(feature_names).index(f) for f in selected_feat] # 선택된 피처들의 인덱스
    rfe.fit(X_proc[:, idx_sel], y_train) # RFE 실행
    selected_feat = [f for f, keep in zip(selected_feat, rfe.support_) if keep] # RFE가 선택한 피처만 남김

# 5. (선택적) SFS/RFA를 이용한 3차 피처 선택
if USE_RFA and len(selected_feat) > RFA_N_FEATURES:
    rfa_est = LGBMClassifier(random_state=RANDOM_STATE, class_weight="balanced", verbose=-1)
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sfs = SequentialFeatureSelector(rfa_est, n_features_to_select=RFA_N_FEATURES, direction="forward", scoring="roc_auc", cv=cv_inner, n_jobs=-1)
    idx_sel = [list(feature_names).index(f) for f in selected_feat]
    sfs.fit(X_proc[:, idx_sel], y_train) # SFS 실행
    selected_feat = [f for f, keep in zip(selected_feat, sfs.get_support()) if keep] # SFS가 선택한 피처만 남김

print(f"[INFO] 최종 선택된 feature 수: {len(selected_feat)}")

# 최종 선택된 피처 이름들을 원래의 컬럼 이름으로 복원
raw_num_selected, raw_cat_selected = [], []
for f in selected_feat:
    if f.startswith("num__"):
        raw_num_selected.append(f.split("num__")[-1])
    elif f.startswith("cat__"):
        col = f.split("cat__")[-1].split("_")[0]
        if col not in raw_cat_selected: raw_cat_selected.append(col)
raw_features = raw_num_selected + raw_cat_selected
print(f"[INFO] 원시 입력 컬럼 수: {len(raw_features)} (num={len(raw_num_selected)}, cat={len(raw_cat_selected)})")

# 선택된 피처들만으로 구성된 새로운 전처리기(Preprocessor)를 만드는 함수
def make_preprocessor_selected():
    transformers = []
    if len(raw_num_selected) > 0: transformers.append(("num", StandardScaler(), raw_num_selected))
    if len(raw_cat_selected) > 0: transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), raw_cat_selected))
    # 만약 선택된 피처가 하나도 없을 경우를 대비한 예외처리
    if not transformers: transformers.append(("num", StandardScaler(), []))
    return ColumnTransformer(transformers=transformers, remainder="drop")

# ==============================================================================
# 5. Optuna를 이용한 하이퍼파라미터 튜닝
# ==============================================================================
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
    """Optuna의 각 trial에서 호출될 목적 함수입니다. 제안된 하이퍼파라미터로 모델을 만들고 교차검증 점수를 반환합니다."""
    try:
        model = build_model(trial, model_name)
        pre_sel = make_preprocessor_selected() # 선택된 피처용 전처리기
        pipeline = Pipeline(steps=[("preprocessor", pre_sel), ("classifier", model)])
        
        # StratifiedGroupKFold: 동일한 사람(group)의 데이터가 train/validation fold에 동시에 들어가지 않도록 분할
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, groups=groups_train,
            scoring=OBJECTIVE_SCORER,
            error_score='raise' # 에러 발생 시 즉시 중단
        )
        return scores.mean() # 3-fold 교차 검증 점수의 평균을 반환
    except Exception as e:
        print(f"ERROR in objective function with model: {model_name}, Error: {e}")
        raise optuna.exceptions.TrialPruned() # 에러 발생 시 해당 trial을 중단

# --- Optuna 실행 루프 ---
optuna_rows, best_models, best_params_by_model = [], {}, {}

for model_name in MODEL_LIST:
    print(f"\n[Optuna] Optimizing {model_name} ...")
    study = optuna.create_study(direction="maximize") # 목표 점수(AUC)를 최대화하는 방향으로 탐색
    study.optimize(lambda trial: objective(trial, model_name), n_trials=N_TRIALS) # N_TRIALS 만큼 최적화 실행
    
    # 최적화 결과 저장
    best_params = study.best_params
    best_score = study.best_value
    print(f"[Optuna] Best {model_name} {OBJECTIVE_METRIC.upper()}={best_score:.3f}, params={best_params}")
    optuna_rows.append({"model": model_name, "objective": OBJECTIVE_METRIC, "best_objective": best_score, "best_params": best_params})
    best_params_by_model[model_name] = best_params
    
    # 찾은 최적 파라미터로 모델을 다시 만들어 학습 데이터 전체로 학습시킨 후 저장
    final_model = build_model(optuna.trial.FixedTrial(best_params), model_name)
    pre_sel = make_preprocessor_selected()
    pipe = Pipeline(steps=[("preprocessor", pre_sel), ("classifier", final_model)])
    pipe.fit(X_train, y_train)
    best_models[model_name] = pipe

# Optuna 튜닝 결과 요약본을 CSV 파일로 저장
pd.DataFrame(optuna_rows).to_csv(os.path.join(ARTIFACT_DIR, "optuna_results.csv"), index=False)


# ==============================================================================
# 6. Hold-out Test: 최적화된 모든 모델을 테스트 데이터로 최종 평가
# ==============================================================================
test_rows = []
per_model_test_preds = {}
for model_name, pipe in best_models.items():
    y_prob = pipe.predict_proba(X_test)[:, 1] # 양성(1) 클래스에 대한 예측 확률
    y_pred = pipe.predict(X_test) # 기본 임계값(0.5) 기준 예측 결과
    
    # 각종 성능 지표 계산
    test_rows.append({
        "model": model_name, "AUC": roc_auc_score(y_test, y_prob),
        "F1": f1_score(y_test, y_pred), "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred), "Accuracy": accuracy_score(y_test, y_pred)
    })
    per_model_test_preds[model_name] = (y_prob, y_pred)

# 테스트 결과를 AUC 기준으로 내림차순 정렬하여 CSV로 저장
test_df = pd.DataFrame(test_rows).sort_values("AUC", ascending=False)
test_df.to_csv(os.path.join(ARTIFACT_DIR, "test_results.csv"), index=False)
print(f"[INFO] Saved ALL model hold-out test results to {os.path.join(ARTIFACT_DIR, 'test_results.csv')}")

# 모델별 성능 지표를 딕셔너리 형태로 저장
holdout_metrics_map = {row['model']: {'AUC': float(row['AUC']), 'F1': float(row['F1']), 'Recall': float(row['Recall']), 'Precision': float(row['Precision']), 'Accuracy': float(row['Accuracy'])} for _, row in test_df.iterrows()}


# ==============================================================================
# 7. 최종 모델 선정 및 임계값 튜닝, 산출물 저장
# ==============================================================================
# 승자 결정: 테스트 데이터에서 AUC 값이 가장 높은 모델
best_model_name = test_df.loc[test_df["AUC"].idxmax(), "model"]
print("-" * 50)
print(f"🏆 [WINNER] Best Model by AUC: {best_model_name} (AUC = {test_df.iloc[0]['AUC']:.4f})")
print("-" * 50)

# --- 최고 모델에 대한 사람 단위 임계값 튜닝 ---
# 1. 사람 단위 예측을 위한 데이터 준비
y_prob_test, _ = per_model_test_preds[best_model_name]
y_true_person = pd.Series(y_test.values, index=groups_test).groupby(level=0).max()
p_person = noisyor_group_probs(groups_test.values, y_prob_test)

# 2. 다양한 전략으로 최적 임계값 탐색
tuned_recall = tune_threshold_noisyor(y_true_person.values, p_person.values, target_recall=TARGET_RECALL)
tuned_youden = tune_threshold_youden(y_true_person.values, p_person.values)
tuned_f1 = tune_threshold_by_metric(y_true_person.values, p_person.values, f1_score, "f1")
tuned_accuracy = tune_threshold_by_metric(y_true_person.values, p_person.values, accuracy_score, "accuracy")

# 튜닝 결과 출력
print(f"[TUNE-Recall]   {best_model_name}: thr={tuned_recall['threshold']:.4f}, recall={tuned_recall['recall']:.3f}, f1={tuned_recall['f1']:.3f}")
print(f"[TUNE-Youden]    {best_model_name}: thr={tuned_youden['threshold']:.4f}, sens={tuned_youden['sensitivity']:.3f}, spec={tuned_youden['specificity']:.3f}, J={tuned_youden['youden']:.3f}")
print(f"[TUNE-F1]        {best_model_name}: thr={tuned_f1['threshold']:.4f}, f1={tuned_f1['f1']:.3f}")
print(f"[TUNE-Accuracy] {best_model_name}: thr={tuned_accuracy['threshold']:.4f}, accuracy={tuned_accuracy['accuracy']:.3f}")

# --- 최종 모델을 전체 데이터로 재학습 ---
params = best_params_by_model[best_model_name]
model = build_model(optuna.trial.FixedTrial(params), best_model_name)
pre_sel_full = make_preprocessor_selected()
pipe_full = Pipeline(steps=[("preprocessor", pre_sel_full), ("classifier", model)])
pipe_full.fit(X_all, y) # 전체 데이터(X_all, y)로 최종 학습

# --- 최종 산출물(Artifact) 생성 ---
# 실제 서비스에서 필요한 모든 정보를 하나의 딕셔너리로 묶음
artifact = {
    "pipeline": pipe_full, # 전체 데이터로 재학습된 최종 모델 파이프라인
    "model_name": best_model_name, # 최고 성능 모델의 이름
    "selected_features": selected_feat, # 전처리 후 사용된 피처 이름 목록
    "raw_features": raw_features, # 원본 데이터에서 사용된 컬럼 이름 목록
    "person_threshold": float(tuned_youden["threshold"]), # 기본으로 사용할 사람 단위 임계값 (Youden's Index 기준)
    "hand_threshold": float(HAND_THRESHOLD_DEFAULT),
    "threshold_strategy": "youden", # 기본 임계값 선택 전략
    "metrics_all": { # 모든 성능 지표 및 임계값 튜닝 결과 저장
        "holdout_test": holdout_metrics_map[best_model_name],
        "recall": tuned_recall,
        "youden": tuned_youden,
        "f1": tuned_f1,
        "accuracy": tuned_accuracy
    }
}

# 최종 아티팩트를 joblib 파일로 저장
save_path = os.path.join(ARTIFACT_DIR, f"best_pipeline_auc_{best_model_name}.joblib")
dump(artifact, save_path)
print(f"\n[SUCCESS] 최종 아티팩트 저장 완료 → {save_path}")
print("[DONE] Training + Threshold tuning + Artifact export complete.")