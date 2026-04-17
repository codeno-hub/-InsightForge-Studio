from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import warnings

warnings.simplefilter("ignore", ConvergenceWarning)

try:
    from imblearn.over_sampling import SMOTE
    _HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    SMOTE = None  # type: ignore
    _HAS_IMBLEARN = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore
    _HAS_MLFLOW = False

TaskType = Literal["classification", "regression"]


@dataclass(frozen=True)
class ModelRunResult:
    model_name: str
    task_type: TaskType
    metrics: Dict[str, float]


@dataclass(frozen=True)
class ModelSelectionResult:
    task_type: TaskType
    target: str
    results: List[ModelRunResult]
    best_model_name: str
    best_pipeline: Pipeline
    input_columns: List[str]
    feature_names: List[str]
    mlflow_run_id: Optional[str] = None


def infer_task_type(y: pd.Series) -> TaskType:
    y_non_null = y.dropna()
    if y_non_null.empty:
        return "classification"

    if pd.api.types.is_numeric_dtype(y_non_null):
        unique_count = int(y_non_null.nunique())
        if unique_count <= 20:
            return "classification"
        return "regression"

    return "classification"


def _detect_skewed_numeric(X: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    skewed: List[str] = []
    for col in numeric_cols:
        values = X[col].dropna()
        if values.empty:
            continue
        if abs(values.skew()) >= threshold:
            skewed.append(col)
    return skewed


def _safe_log_transform(array: np.ndarray) -> np.ndarray:
    return np.sign(array) * np.log1p(np.abs(array))


def _build_preprocessor(X: pd.DataFrame, log_transform: bool = False) -> ColumnTransformer:
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if log_transform and numeric_cols:
        numeric_steps.append(("log", FunctionTransformer(_safe_log_transform, validate=False)))
    numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def _get_feature_names(preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names: List[str] = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue

            if isinstance(transformer, Pipeline):
                transformer = transformer.steps[-1][1]

            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names = transformer.get_feature_names_out(cols)
                except Exception:
                    names = [f"{name}__{c}" for c in cols]
            else:
                names = [f"{name}__{c}" for c in cols]

            feature_names.extend(names)
        return feature_names


def _apply_smote(X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, pd.Series]:
    if not _HAS_IMBLEARN:
        return X, y

    try:
        return SMOTE(random_state=42).fit_resample(X, y)
    except Exception:
        return X, y


def _get_class_imbalance_params(y: pd.Series) -> Tuple[Optional[str], float]:
    if y.nunique() == 2:
        counts = y.value_counts()
        major, minor = counts.iloc[0], counts.iloc[-1]
        if minor == 0:
            return "balanced", 1.0
        ratio = major / minor
        return ("balanced", float(ratio)) if ratio >= 2.0 else (None, float(ratio))

    return (None, 1.0)


def _get_models(task_type: TaskType, class_weight: Optional[str] = None, scale_pos_weight: float = 1.0) -> List[Tuple[str, Any]]:
    models: List[Tuple[str, Any]] = []

    if task_type == "classification":
        models.append(
            (
                "Logistic Regression",
                LogisticRegression(
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=1000,
                    class_weight=class_weight,
                ),
            )
        )
        models.append(
            (
                "Random Forest",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=42,
                    class_weight=class_weight,
                    n_jobs=-1,
                ),
            )
        )
        try:
            from xgboost import XGBClassifier

            params: Dict[str, Any] = {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_lambda": 1.0,
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "logloss",
            }
            if scale_pos_weight > 1.0:
                params["scale_pos_weight"] = scale_pos_weight

            models.append(("XGBoost", XGBClassifier(**params)))
        except Exception:
            pass
    else:
        models.append(
            (
                "Random Forest",
                RandomForestRegressor(
                    n_estimators=400,
                    random_state=42,
                    n_jobs=-1,
                ),
            )
        )
        try:
            from xgboost import XGBRegressor

            models.append(
                (
                    "XGBoost",
                    XGBRegressor(
                        n_estimators=600,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=1.0,
                        random_state=42,
                        n_jobs=-1,
                    ),
                )
            )
        except Exception:
            pass

    return models


def _build_search_grid(model_name: str, task_type: TaskType) -> Dict[str, List[Any]]:
    if model_name == "Logistic Regression":
        return {
            "model__C": [0.1, 1.0, 10.0],
            "model__solver": ["liblinear"],
            "model__penalty": ["l2"],
        }

    if model_name == "Random Forest":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
        }

    if model_name == "XGBoost":
        return {
            "model__n_estimators": [100, 200],
            "model__max_depth": [4, 6],
            "model__learning_rate": [0.05, 0.1],
        }

    return {}


def _optimize_pipeline(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    task_type: TaskType,
) -> Tuple[Pipeline, Dict[str, Any]]:
    param_grid = _build_search_grid(model_name, task_type)
    if not param_grid:
        return pipe, {}

    scoring = "f1_weighted" if task_type == "classification" else "r2"
    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def _fit_pipeline(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_smote: bool,
    task_type: TaskType,
) -> Tuple[Pipeline, np.ndarray]:
    if use_smote and task_type == "classification" and _HAS_IMBLEARN:
        preprocessor = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_train_resampled, y_train_resampled = _apply_smote(X_train_transformed, y_train)
        model.fit(X_train_resampled, y_train_resampled)
        return pipe, X_train_transformed

    pipe.fit(X_train, y_train)
    X_train_transformed = pipe.named_steps["preprocess"].transform(X_train)
    return pipe, X_train_transformed


def _safe_roc_auc(y_true: np.ndarray, y_proba: Optional[np.ndarray]) -> Optional[float]:
    if y_proba is None:
        return None
    try:
        if len(np.unique(y_true)) != 2:
            return None
        return float(roc_auc_score(y_true, y_proba))
    except Exception:
        return None


def _evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    auc = _safe_roc_auc(y_true, y_proba)
    if auc is not None:
        metrics["roc_auc"] = float(auc)
    return metrics


def _evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _get_model_metrics(
    pipe: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    task_type: TaskType,
) -> Dict[str, float]:
    y_pred = pipe.predict(X_test)
    if task_type == "classification":
        y_true = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        y_proba: Optional[np.ndarray] = None
        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_proba = np.asarray(proba[:, 1])
            except Exception:
                y_proba = None
        return _evaluate_classification(y_true, y_pred_arr, y_proba)

    return _evaluate_regression(np.asarray(y_test), np.asarray(y_pred))


def _log_mlflow_run(
    task_type: TaskType,
    target_col: str,
    test_size: float,
    optimize_model_name: Optional[str],
    imbalance_ratio: float,
    results: List[ModelRunResult],
    best_model_name: str,
    best_pipeline: Pipeline,
) -> Optional[str]:
    if not _HAS_MLFLOW:
        return None

    try:
        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("AutoEDA Model Selector")
        with mlflow.start_run(run_name=f"{task_type.capitalize()} - {target_col}") as run:
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("target_column", target_col)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("optimize_model", optimize_model_name or "none")
            mlflow.log_param("imbalance_ratio", imbalance_ratio)
            mlflow.log_param("best_model", best_model_name)
            for model_result in results:
                for metric_name, metric_value in model_result.metrics.items():
                    mlflow.log_metric(f"{model_result.model_name}_{metric_name}", metric_value)
            try:
                mlflow.sklearn.log_model(best_pipeline, "best_pipeline")
            except Exception:
                pass
            return run.info.run_id
    except Exception:
        return None


def train_and_compare_models(
    df: pd.DataFrame,
    *,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    optimize: bool = False,
    optimize_model_name: Optional[str] = None,
    use_log_transform: bool = False,
    use_smote: bool = False,
    track_experiment: bool = False,
) -> ModelSelectionResult:
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    task_type = infer_task_type(y)

    mask = ~y.isna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    if X.empty:
        raise ValueError("No training rows available after dropping missing targets.")

    imbalance_label, imbalance_ratio = _get_class_imbalance_params(y) if task_type == "classification" else (None, 1.0)
    preprocessor = _build_preprocessor(X, log_transform=use_log_transform)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == "classification" else None,
    )

    class_weight = imbalance_label
    scale_pos_weight = imbalance_ratio if task_type == "classification" else 1.0
    candidate_models = _get_models(task_type, class_weight=class_weight, scale_pos_weight=scale_pos_weight)

    results: List[ModelRunResult] = []
    trained_pipelines: Dict[str, Pipeline] = {}
    mlflow_run_id: Optional[str] = None

    for model_name, model in candidate_models:
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        if optimize and optimize_model_name == model_name:
            trained_pipeline, _ = _optimize_pipeline(pipe, X_train, y_train, model_name, task_type)
            model_name = f"{model_name} (Optimized)"
        else:
            trained_pipeline, _ = _fit_pipeline(pipe, X_train, y_train, use_smote=use_smote, task_type=task_type)

        metrics = _get_model_metrics(trained_pipeline, X_test, y_test, task_type)
        results.append(ModelRunResult(model_name=model_name, task_type=task_type, metrics=metrics))
        trained_pipelines[model_name] = trained_pipeline

    if task_type == "classification":
        best = max(results, key=lambda r: r.metrics.get("f1", float("-inf")))
    else:
        best = max(results, key=lambda r: r.metrics.get("r2", float("-inf")))

    best_pipeline = trained_pipelines.get(best.model_name)
    if best_pipeline is None:
        best_pipeline = next(iter(trained_pipelines.values()))

    if track_experiment:
        mlflow_run_id = _log_mlflow_run(
            task_type=task_type,
            target_col=target_col,
            test_size=test_size,
            optimize_model_name=optimize_model_name,
            imbalance_ratio=imbalance_ratio,
            results=results,
            best_model_name=best.model_name,
            best_pipeline=best_pipeline,
        )

    return ModelSelectionResult(
        task_type=task_type,
        target=target_col,
        results=results,
        best_model_name=best.model_name,
        best_pipeline=best_pipeline,
        input_columns=list(X.columns),
        feature_names=_get_feature_names(preprocessor, X),
        mlflow_run_id=mlflow_run_id,
    )

