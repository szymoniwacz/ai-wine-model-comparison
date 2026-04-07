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
        - Summary table with model ranking, accuracy, difference vs best, and model path
        - Best model(s) highlighted and interpreted
        - Accuracy differences for each model (as %)
        - Confusion matrix and classification report for each model
        - Example output:

```
Model comparison results:
Rank  Model                Accuracy   Δ vs best   Model Path
============================================================
1     logistic_regression  0.9722     +0.00%      artifacts/model_logistic_regression.joblib
2     svm                  0.9722     +0.00%      artifacts/model_svm.joblib
3     decision_tree        0.9444     -2.86%      artifacts/model_decision_tree.joblib

Best model(s): logistic_regression, svm (accuracy: 0.9722)
Multiple models achieved the same best accuracy. You may choose based on speed, interpretability, or other factors.

Interpretation:
- logistic_regression is a top performer on this dataset.
- svm is a top performer on this dataset.
- decision_tree is 2.86% less accurate than the best model.
```

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

## How to interpret results

- **Ranking**: Models are ranked by accuracy. The best model(s) are highlighted.
- **Δ vs best**: Shows how much less accurate each model is compared to the best (in %).
- **Interpretation**: The CLI prints a short summary and recommendation based on the results.
- **Confusion matrix & classification report**: For each model, you get a detailed breakdown of predictions and per-class metrics.

**Tip:** If multiple models have the same accuracy, consider other factors (speed, interpretability, robustness) when choosing your production model.
