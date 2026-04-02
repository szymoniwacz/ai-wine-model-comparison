# AI Wine Model Comparison

Professional project for comparing machine learning models on the wine dataset from scikit-learn.

---

## Project Structure

- `src/` – source code
- `data/` – data (empty by default, gitignored)
- `notebooks/` – experimental notebooks
- `requirements.txt` – dependencies
- `.gitignore` – ignore file

---

## Quick Start

1. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Train your first model:
    ```bash
    python -m src.cli train --model [model_type]
    ```
   The trained model will be saved to `artifacts/model_[model_type].joblib` and accuracy will be printed in the terminal.

---

## Project Goal
Compare various ML models (e.g., decision trees, SVM, logistic regression) on the wine dataset, analyze results, and visualize findings.

---

## CLI Commands

- `python -m src.cli train --model [model_type]` — trains a model of the selected type. Output: accuracy and model path.
    - `[model_type]` can be one of:
        - `decision_tree` (default)
        - `logistic_regression`
        - `svm`
    - Example:
        ```bash
        python -m src.cli train --model decision_tree
        python -m src.cli train --model logistic_regression
        python -m src.cli train --model svm
        ```
    - Each model is saved to a separate file: `artifacts/model_[model_type].joblib`

---

## Model Training

- Supported models:
    - Decision Tree (with max_depth=3)
    - Logistic Regression (with StandardScaler)
    - SVM (with StandardScaler)
- Model is trained on the wine dataset from scikit-learn
- Model and class names are saved to `artifacts/model_[model_type].joblib` using joblib
- Accuracy is evaluated on the test set and printed after training
