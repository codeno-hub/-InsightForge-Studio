# InsightForge Studio

A Streamlit-powered AI workspace for automated EDA, feature engineering, model training, explainability, and batch prediction.

## 🚀 Features

- **📁 Data Upload**: Support for CSV (`.csv`) and Parquet (`.parquet`, `.pq`) files
- **📊 Auto-EDA**: Generates comprehensive HTML reports using `ydata-profiling`
- **🤖 Model Studio**:
  - **Classification**: Logistic Regression, Random Forest, XGBoost
  - **Regression**: Random Forest, XGBoost
  - Automatic preprocessing with imputation, scaling, and categorical encoding
  - Optional log-transform for skewed numerical features
  - Simple train/test split evaluation with metrics display
  - Feature exclusion to prevent data leakage

## 📂 Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml         # GitHub Actions CI configuration
├── app.py                 # Main Streamlit application
├── eda_engine.py          # EDA report generation engine
├── model_selector.py      # ML model training and comparison
├── pyproject.toml         # Python project configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
└── .gitignore             # Git ignore rules
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

   *Replace `your-username` and `your-repo-name` with your actual GitHub username and repository name.*

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ✨ Professional upgrades

This version of the app now includes:

- Automatic feature engineering with standard scaling and optional log-transform for skewed numeric features
- Class imbalance handling with SMOTE support and balanced class weights for classification tasks
- Model explainability via feature importance charts and SHAP value visualizations
- Hyperparameter optimization using quick GridSearchCV for a selected model
- Experiment tracking with MLflow and local `mlruns` storage
- Model export as a `.pkl` pipeline and batch prediction support for new datasets
- Docker-ready deployment for cloud and container workflows

## 🚀 Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then follow these steps in the web interface:

1. **Upload a dataset**: Choose a CSV or Parquet file
2. **Preview data**: View the first 50 rows and dataset shape
3. **Generate EDA report** (optional): Click to create and download an HTML report
4. **Select target column**: Choose the column to predict
5. **Train models**: Click to train and compare baseline models

### Run with Docker

Build the container image:

```bash
docker build -t insightforge-studio .
```

Run the app locally:

```bash
docker run -p 8501:8501 insightforge-studio
```

### MLflow tracking

Experiment results are logged to a local `mlruns/` directory when training runs complete.

## � Continuous Integration

This project uses GitHub Actions for automated testing. The CI pipeline:

- Tests the application on multiple Python versions (3.8 through 3.13)
- Verifies that all modules can be imported successfully
- Runs on every push and pull request to the main branch

Check the [Actions tab](https://github.com/your-username/your-repo-name/actions) in your GitHub repository to see the CI status.

## �📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

**Note on Parquet files**: If you encounter issues loading Parquet files, install `pyarrow`:

```bash
pip install pyarrow
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Notes

- Generated EDA reports are saved to `.streamlit_artifacts/` and are ignored by Git.
- The app performs basic preprocessing; for production use, consider more advanced feature engineering.

