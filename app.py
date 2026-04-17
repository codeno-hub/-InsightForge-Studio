from __future__ import annotations

import importlib.util
import pickle
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from eda_engine import generate_eda_report_html
from model_selector import (
    ModelSelectionResult,
    infer_task_type,
    train_and_compare_models,
)


@st.cache_data(show_spinner=False)
def _cached_eda_report(
    df: pd.DataFrame,
    minimal: bool,
    title: str,
    filename: str,
) -> bytes:
    return generate_eda_report_html(df, title=title, minimal=minimal)


def _read_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    raise ValueError("Unsupported file type. Please upload a CSV or Parquet file.")


def _results_to_dataframe(sel: ModelSelectionResult) -> pd.DataFrame:
    rows = []
    for r in sel.results:
        row = {"model": r.model_name, **r.metrics}
        rows.append(row)
    df = pd.DataFrame(rows)
    if sel.task_type == "classification":
        sort_key = "f1" if "f1" in df.columns else None
        if sort_key:
            df = df.sort_values(sort_key, ascending=False)
    else:
        sort_key = "r2" if "r2" in df.columns else None
        if sort_key:
            df = df.sort_values(sort_key, ascending=False)
    return df.reset_index(drop=True)


def _get_feature_importance(
    pipeline: Any,
    feature_names: List[str],
) -> Optional[pd.DataFrame]:
    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        importances = np.mean(np.abs(coef), axis=0)

    if importances is None or len(importances) != len(feature_names):
        return None

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    return importance_df.head(20)


def _render_shap_explanation(
    pipeline: Any,
    X: pd.DataFrame,
    feature_names: List[str],
) -> Optional[bytes]:
    try:
        import shap
    except ImportError:
        return None

    try:
        preprocessor = pipeline.named_steps["preprocess"]
        model = pipeline.named_steps["model"]
        X_transformed = preprocessor.transform(X)
        explainer = shap.Explainer(model, X_transformed, feature_names=feature_names)
        shap_values = explainer(X_transformed[:200])

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_transformed[:200], feature_names=feature_names, show=False)
        plt.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None


def _is_skewed(df: pd.DataFrame, threshold: float = 1.0) -> List[str]:
    skewed_columns = []
    for column in df.select_dtypes(include=["number"]):
        values = df[column].dropna()
        if values.empty:
            continue
        if abs(values.skew()) >= threshold:
            skewed_columns.append(column)
    return skewed_columns


def _make_download_button(pipeline: Any, model_name: str) -> None:
    try:
        payload = pickle.dumps(pipeline)
    except Exception:
        st.warning("Model export is not available for this pipeline.")
        return

    st.download_button(
        label="Download trained model (.pkl)",
        data=payload,
        file_name=f"{model_name.replace(' ', '_')}.pkl",
        mime="application/octet-stream",
    )


def main() -> None:
    st.set_page_config(page_title="InsightForge Studio", layout="wide")
    st.title("InsightForge Studio")
    st.caption(
        "Upload a dataset, generate cached EDA reports, train models, and export production-ready predictions."
    )

    if "trained_pipeline" not in st.session_state:
        st.session_state.trained_pipeline = None
        st.session_state.input_columns = []
        st.session_state.feature_names = []
        st.session_state.target_column = None

    with st.sidebar:
        st.header("Dataset & training settings")
        uploaded = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet", "pq"])
        df = None
        if uploaded:
            try:
                df = _read_uploaded_file(uploaded)
            except Exception as exc:
                st.error(f"Unable to read file: {exc}")

        if df is not None:
            st.markdown(
                f"**Rows:** {df.shape[0]:,}  \n**Columns:** {df.shape[1]:,}"
            )
            exclude_columns = st.multiselect(
                "Select columns to drop",
                options=list(df.columns),
                key="exclude_columns",
            )
            target_col = st.selectbox("Target column", options=list(df.columns), key="target_col")
            test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            if target_col:
                training_df = df.drop(columns=[col for col in exclude_columns if col != target_col], errors="ignore")
                if target_col in exclude_columns:
                    st.warning("The selected target column was included in the excluded columns list and will not be dropped.")
                task_type = infer_task_type(training_df[target_col])
                task_type = infer_task_type(df[target_col])
                st.markdown(f"**Detected task:** {task_type}")
                skewed = _is_skewed(df.drop(columns=[target_col]))
                if skewed:
                    st.info(
                        "Skewed numeric features detected: "
                        + ", ".join(skewed[:5])
                        + ("..." if len(skewed) > 5 else "")
                    )
                use_log_transform = st.checkbox(
                    "Apply log-transform for skewed numeric features",
                    value=False,
                )
                use_smote = False
                if task_type == "classification":
                    use_smote = st.checkbox(
                        "Use SMOTE for class imbalance (if available)",
                        value=False,
                    )
                optimize = st.checkbox("Enable model optimization", value=False)
                optimize_model_name = None
                if optimize:
                    xgboost_installed = importlib.util.find_spec("xgboost") is not None
                    if task_type == "classification":
                        model_options = ["Logistic Regression", "Random Forest"]
                        if xgboost_installed:
                            model_options.append("XGBoost")
                    else:
                        model_options = ["Random Forest"]
                        if xgboost_installed:
                            model_options.append("XGBoost")

                    optimize_model_name = st.selectbox(
                        "Select model to optimize",
                        options=model_options,
                        index=0,
                        key="optimize_model",
                    )
                train_btn = st.button("Train models and evaluate")
            else:
                st.warning("Choose a target column to continue.")
                train_btn = False
        else:
            target_col = None
            test_size = 0.2
            use_log_transform = False
            use_smote = False
            optimize = False
            optimize_model_name = None
            train_btn = False

    if df is None:
        st.info("Upload a dataset from the sidebar to start.")
        return

    tabs = st.tabs(["Overview", "Auto-EDA", "Model training", "Batch prediction"])

    with tabs[0]:
        st.subheader("Data preview")
        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        st.dataframe(df.head(50), use_container_width=True)

    with tabs[1]:
        st.subheader("Auto-EDA report")
        minimal = st.checkbox("Minimal report (faster)", value=True, key="eda_minimal")
        report_name = st.text_input("Report filename", value="eda_report.html", key="eda_filename")
        if st.button("Generate EDA report", type="primary", key="generate_report"):
            with st.spinner("Generating EDA report..."):
                try:
                    report_bytes = _cached_eda_report(
                        df,
                        minimal=minimal,
                        title=f"EDA Report: {uploaded.name}",
                        filename=report_name.strip() or "eda_report.html",
                    )
                    output_dir = Path(".streamlit_artifacts")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    report_path = output_dir / (report_name.strip() or "eda_report.html")
                    report_path.write_bytes(report_bytes)
                    st.success("EDA report generated and cached.")
                    st.download_button(
                        label="Download EDA HTML report",
                        data=report_bytes,
                        file_name=report_path.name,
                        mime="text/html",
                    )
                    st.caption(f"Saved to: {report_path}")
                except Exception as exc:
                    st.error(f"EDA report generation failed: {exc}")

    with tabs[2]:
        st.subheader("Model training and evaluation")
        if train_btn:
            with st.spinner("Training and evaluating models..."):
                try:
                    selection = train_and_compare_models(
                        training_df,
                        target_col=target_col,
                        test_size=float(test_size),
                        optimize=optimize,
                        optimize_model_name=optimize_model_name,
                        use_log_transform=use_log_transform,
                        use_smote=use_smote,
                        track_experiment=True,
                    )
                except Exception as exc:
                    st.error(f"Model training failed: {exc}")
                    return

            st.session_state.trained_pipeline = selection.best_pipeline
            st.session_state.input_columns = selection.input_columns
            st.session_state.feature_names = selection.feature_names
            st.session_state.target_column = selection.target

            st.success(
                f"Task inferred: {selection.task_type}. Best model: {selection.best_model_name}."
            )
            if selection.mlflow_run_id:
                st.info(f"Experiment logged to MLflow run: {selection.mlflow_run_id}")

            results_df = _results_to_dataframe(selection)
            st.dataframe(results_df, use_container_width=True)

            importance_df = _get_feature_importance(selection.best_pipeline, selection.feature_names)
            if importance_df is not None:
                st.subheader("Feature importance")
                st.bar_chart(importance_df.set_index("feature")["importance"])

            _make_download_button(selection.best_pipeline, selection.best_model_name)

            shap_image = _render_shap_explanation(
                selection.best_pipeline,
                df.drop(columns=[selection.target]),
                selection.feature_names,
            )
            with st.expander("Model explainability (SHAP)"):
                if shap_image is None:
                    st.info(
                        "Install the 'shap' package to display feature impact plots for the selected model."
                    )
                else:
                    st.image(shap_image, use_column_width=True)
        else:
            st.info("Train a model using the sidebar controls.")

    with tabs[3]:
        st.subheader("Batch prediction")
        if st.session_state.trained_pipeline is None:
            st.warning("Train a model before using batch prediction.")
        else:
            batch_file = st.file_uploader(
                "Upload new data for batch predictions",
                type=["csv", "parquet", "pq"],
                key="batch_upload",
            )
            if batch_file is not None:
                try:
                    batch_df = _read_uploaded_file(batch_file)
                except Exception as exc:
                    st.error(f"Unable to read batch file: {exc}")
                    return

                missing_columns = [
                    col for col in st.session_state.input_columns if col not in batch_df.columns
                ]
                if missing_columns:
                    st.error(
                        "Batch input is missing required columns: "
                        + ", ".join(missing_columns[:10])
                    )
                    return

                predictors = batch_df[st.session_state.input_columns].copy()
                predictions = st.session_state.trained_pipeline.predict(predictors)
                prediction_name = "prediction"
                batch_df[prediction_name] = predictions
                st.dataframe(batch_df.head(50), use_container_width=True)
                csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download batch predictions",
                    data=csv_bytes,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()

