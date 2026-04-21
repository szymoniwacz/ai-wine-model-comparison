import os
import sys
from sklearn.datasets import load_wine
from src.trainer import train
from src.available_models import AVAILABLE_MODELS
from src.formatters.confusion_matrix_formatter import format_confusion_matrix_result
from src.formatters.comparison_formatter import (
    print_comparison_table,
    print_model_reports,
)


def print_usage():
    print("Usage:")
    print("  python -m src.cli train [--model MODEL_TYPE]")
    print("    --model MODEL_TYPE: decision_tree (default), logistic_regression, svm")
    print("  python -m src.cli compare")
    print("  python -m src.cli experiment-model-behavior")


def print_model_result(model_type, result):
    print(
        f"Model: {model_type} | Accuracy: {result['accuracy']:.4f} | Saved to: {result['model_path']}"
    )
    wine = load_wine()
    formatter_input = {
        "matrix": result["confusion_matrix"],
        "labels": wine.target_names.tolist(),
        "plot_path": "(not generated)",
        "classification_report": "(not implemented)",
    }
    print(format_confusion_matrix_result(formatter_input))
    print()


def handle_train(args):
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            model_type = args[idx + 1]
            print(f"Training model: {model_type}. Please wait...\n")
            result = train(model_type=model_type)
            print_model_result(model_type, result)
        else:
            print("Error: --model provided but no model type specified.")
    else:
        print(
            f"No --model specified. Training all models: {', '.join(AVAILABLE_MODELS)}. Please wait..."
        )
        for model_type in AVAILABLE_MODELS:
            print(f"-" * 120)
            print(f"Training model: {model_type}. Please wait...\n")
            result = train(model_type=model_type)
            print_model_result(model_type, result)
            print(f"-" * 120) if model_type == AVAILABLE_MODELS[-1] else ""
        print("All models trained.\n")


def handle_compare(args):
    from src.trainer import compare_models

    print("Comparing all models on the same test set...\n")
    results = compare_models()
    print_comparison_table(results)

    # Plot model accuracies
    try:
        import matplotlib.pyplot as plt

        model_names = [r["model"] for r in results]
        accuracies = [r["accuracy"] for r in results]
        plt.figure(figsize=(8, 5))
        bars = plt.bar(model_names, accuracies, color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy Comparison")
        for bar, acc in zip(bars, accuracies):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )
        os.makedirs("artifacts", exist_ok=True)
        plot_path = os.path.join("artifacts", "model_comparison.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"\nAccuracy bar plot saved to: {plot_path}\n")
    except ImportError:
        print(
            "matplotlib is not installed. Skipping plot generation. To enable, install matplotlib."
        )

    wine = load_wine()
    print_model_reports(results, wine)


def handle_experiment_model_behavior(_args: list) -> None:
    from src.experiments.model_behavior import run

    print(
        "Running experiment: model behavior comparison (Logistic Regression vs SVM)...\n"
    )
    result = run()

    print("Accuracies:")
    for name, acc in result["accuracies"].items():
        print(f"  {name.replace('_', ' ').title()}: {acc:.4f}")

    print(
        f"\nDisagreements: {result['disagreement_count']} test sample(s) "
        "where Logistic Regression and SVM predict differently"
    )

    print("\nGenerated artifacts:")
    for key, path in result["artifact_paths"].items():
        print(f"  [{key}] {path}")


COMMAND_HANDLERS = {
    "train": handle_train,
    "compare": handle_compare,
    "experiment-model-behavior": handle_experiment_model_behavior,
}


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    handler = COMMAND_HANDLERS.get(command)

    if handler is None:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(COMMAND_HANDLERS.keys())}")
        print_usage()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
