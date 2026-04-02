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
2. Train all models (or a specific one):
    ```bash
    # Train all models
    python -m src.cli train

    # Or train a specific model
    python -m src.cli train --model decision_tree
    python -m src.cli train --model logistic_regression
    python -m src.cli train --model svm
    ```
   Each trained model will be saved to `artifacts/model_[model_type].joblib` and accuracy will be printed in the terminal.

---

## Project Goal
Compare various ML models (e.g., decision trees, SVM, logistic regression) on the wine dataset, analyze results, and visualize findings.

---

## CLI Commands

- `python -m src.cli train [--model MODEL_TYPE]` — trains models. Output: accuracy and model path.
    - If `--model` is not provided, all models will be trained sequentially: `decision_tree`, `logistic_regression`, `svm`.
    - If `--model MODEL_TYPE` is provided, only the selected model will be trained.
    - Example:
        ```bash
        python -m src.cli train           # trains all models
        python -m src.cli train --model decision_tree
        python -m src.cli train --model logistic_regression
        python -m src.cli train --model svm
        ```
    - Each model is saved to a separate file: `artifacts/model_[model_type].joblib`

- `python -m src.cli compare` — trains and evaluates all models, then compares their results.
    - Output:
        - Summary table with model name, accuracy, and model path
        - Best model(s) highlighted
        - Accuracy differences for each model
        - Confusion matrix for each model
    - Example:
        ```bash
        python -m src.cli compare
        ```

---

## Model Training

- Supported models:
    - Decision Tree (with max_depth=3)
    - Logistic Regression (with StandardScaler)
    - SVM (with StandardScaler)
- Model is trained on the wine dataset from scikit-learn
- Model and class names are saved to `artifacts/model_[model_type].joblib` using joblib
- Accuracy is evaluated on the test set and printed after training
