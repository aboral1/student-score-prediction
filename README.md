# Student Score Prediction

An educational machine learning project that explores student score prediction through two Jupyter Notebook workflows: regression on a CSV dataset and model comparison with MLflow experiment tracking.

The notebooks cover data exploration, preprocessing, model training, evaluation, and residual analysis. They are intended for learning how to build and compare regression models and how to record experiments for later review.

## Workflows at a Glance

| Workflow | Data | Models | Main focus |
| --- | --- | --- | --- |
| Student Score Prediction | Included student performance CSV | Linear regression and degree-2 polynomial regression | Explore relationships, compare model complexity, and visualize predictions |
| Student Score Prediction with MLflow | Generated synthetic data | Linear regression, Ridge regression, and random forest | Track parameters, metrics, models, and error analysis |

## 1. Student Score Prediction

**Notebook:** [Student Score Prediction.ipynb](<Student Score Prediction.ipynb>)

This workflow loads the included dataset, explores student attributes, and compares a linear regression baseline with a polynomial regression model that can represent quadratic relationships and feature interactions.

### Dataset and Features

The dataset is stored at [`Code/data/Students Performance .csv`](<Code/data/Students Performance .csv>). The filename includes a space before `.csv`; preserve that space when referencing the file.

It contains 145 records and 16 columns, including demographic information, study habits, classroom behavior, grades, and a `Target_Score` column. The notebook uses `Target_Score` as its regression target, copying it into a column named `Score`.

The selected prediction features are:

- `Weekly_Study_Hours`
- `Attendance`
- `Reading`
- `Listening_in_Class`
- `Project_work`

### Processing and Training

1. Load the CSV and inspect its contents.
2. Explore grade distributions, study hours, and target score statistics.
3. Encode categorical variables and convert age ranges into numeric values.
4. Examine correlations between the selected features and the target.
5. Split the data into 80% training and 20% test sets using `random_state=42`.
6. Train a linear regression model.
7. Train a pipeline containing degree-2 polynomial features, standard scaling, and linear regression.
8. Compare evaluation metrics, prediction plots, fitted curves, and residuals.

The comparison illustrates how adding polynomial features changes the model's ability to fit curved patterns. Use test performance and residuals to judge whether the extra complexity is useful.

## 2. Student Score Prediction with MLflow

**Notebook:** [Student Score Prediction System - MLFlow.ipynb](<Student Score Prediction System - MLFlow.ipynb>)

This workflow generates data locally and demonstrates how to train several regressors while recording their results in MLflow.

### Synthetic Data

The notebook uses Scikit-learn's `make_regression` to generate 500 samples with 10 features, of which 8 are informative. It assigns student-related names such as `Study_Hours`, `Previous_GPA`, `Attendance`, and `Assignment_Score` to the generated features.

The generated target is transformed and clipped to the 0–100 range. These feature names and scores are illustrative; the generated values do not represent measured student behavior. Clipping also changes the original regression relationship, so results should be interpreted within this demonstration.

The data is split into 80% training and 20% test sets. `StandardScaler` is fitted on the training features and then used to transform both sets.

### Models and Tracking

The notebook trains four configurations:

| Run | Configuration |
| --- | --- |
| Linear Regression | Baseline linear model |
| Ridge_Alpha_1 | Ridge regression with `alpha=1.0` |
| Ridge_Alpha_10 | Ridge regression with `alpha=10.0` |
| Random_Forest_100 | Random forest with 100 trees and `random_state=42` |

Under the experiment **`Student_Score_Prediction`**, each training run logs model parameters, training and test R², test RMSE, test MAE, and the fitted model.

The notebook selects the model with the highest test R² for error analysis. It creates residual plots and a Q–Q plot, saves `error_distribution.png`, and logs the error distribution image and summary statistics in a separate **`Error Analysis`** run.

The active tracking backend is the local SQLite database `mlflow.db`. The notebook first sets a file tracking URI, then switches to `sqlite:///mlflow.db` during experiment setup.

## Project Structure

```text
.
├── README.md
├── Student Score Prediction.ipynb
├── Student Score Prediction System - MLFlow.ipynb
├── Assignment_Class_2.ipynb
├── Solution_Class_2.ipynb
├── Code/
│   ├── README.md
│   ├── requirements.txt
│   ├── generate_data.ipynb
│   ├── train_model.ipynb
│   ├── data/
│   │   └── Students Performance .csv
│   └── notebooks/
│       └── Student_Score_Prediction_Regression.ipynb
├── Theory/
├── Day2/
├── mlflow.db
├── mlruns/
└── error_distribution.png
```

Start with the two notebooks in the project root. The `Code/` directory contains additional training and data-generation notebooks, while the assignment, solution, and theory files provide supporting class material. MLflow files and the error image are local experiment outputs.

## Technologies

- Python
- Pandas and NumPy for data manipulation and numerical operations
- Scikit-learn for preprocessing, regression, and evaluation
- Matplotlib and Seaborn for visualizations
- SciPy for statistical analysis and diagnostic plots
- MLflow for experiment tracking and model artifacts
- Jupyter Notebook for interactive execution

The existing `Code/requirements.txt` also includes DVC.

## Getting Started

Run the following commands from the project root so that the notebooks resolve their relative dataset and output paths correctly.

### 1. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
python -m pip install -r Code/requirements.txt
python -m pip install scipy notebook ipykernel
```

The second command explicitly installs SciPy and the notebook tools, which are not listed in the existing requirements file.

### 3. Launch Jupyter Notebook

```bash
python -m notebook
```

Open either root notebook and run its cells in order from top to bottom. If using an IDE, select the Python interpreter or notebook kernel associated with `.venv`.

### 4. View MLflow Experiments

After running the MLflow notebook, open a second terminal, activate the same environment, and run this command from the project root:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open [http://localhost:5000](http://localhost:5000) and select the `Student_Score_Prediction` experiment to inspect run metrics, parameters, and artifacts. Additional notebook executions create additional runs in the same experiment.

## Evaluation Metrics

| Metric | Interpretation |
| --- | --- |
| R² | Measures performance relative to predicting the target mean. Higher is better; 1 indicates a perfect fit, and values can be negative. |
| RMSE | Measures prediction error in score units and penalizes large errors more strongly. Lower is better. |
| MAE | Measures average absolute prediction error in score units. Lower is better. |

Compare training and test metrics together. Strong training performance with weaker test performance can indicate overfitting. Residual plots help reveal systematic errors that a single metric can hide.

## Learning Goals and Limitations

After completing the notebooks, you should be able to explore a regression dataset, prepare features, compare several models, interpret evaluation metrics, and inspect MLflow experiment records.

These are teaching workflows. The CSV is small, the MLflow dataset is synthetic, and the models are evaluated using a single train/test split. The CSV workflow also uses label encoding for categorical features, which introduces numeric ordering that may not reflect their meaning. Before applying these approaches to real student data, revisit the target definition, encoding strategy, and validation design.

## Troubleshooting

- **CSV:** Start Jupyter from the project root and check the exact path `data/Students Performance .csv`.
- **`Target_Score` missing:** Use the included CSV; the first notebook explicitly checks for this column.
- **Import errors:** Confirm that the selected notebook kernel uses `.venv` and that both dependency installation commands completed.
- **No MLflow runs visible:** Run the training cells and launch the UI from the same directory using `sqlite:///mlflow.db`.
- **Port 5000 already in use:** Launch the UI with `--port 5001` and open `http://localhost:5001`.
- **Variables are undefined:** Restart the kernel and run all preceding cells in order.
