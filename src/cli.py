import sys
from sklearn.datasets import load_wine
from src.trainer import train
from src.available_models import AVAILABLE_MODELS
from src.formatters.confusion_matrix_formatter import format_confusion_matrix_result


def print_usage():
    print("Usage:")
    print("  python -m src.cli train [--model MODEL_TYPE]")
    print("    --model MODEL_TYPE: decision_tree (default), logistic_regression, svm")
    # Future: add more commands here


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


def print_comparison_table(results):
    print("Model comparison results:")
    header = f"{'Model':<20} {'Accuracy':<10} {'Model Path'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['model']:<20} {r['accuracy']:<10.4f} {r['model_path']}")
    print()

    # Find best model(s)
    best_acc = max(r["accuracy"] for r in results)
    best_models = [r["model"] for r in results if r["accuracy"] == best_acc]
    print(f"Best model(s): {', '.join(best_models)} (accuracy: {best_acc:.4f})")
    print()
    # Show accuracy differences
    print("Accuracy differences:")
    for r in results:
        diff = best_acc - r["accuracy"]
        print(f"{r['model']:<20} {diff:+.4f}")
    print()


def handle_compare(args):
    from src.trainer import compare_models

    print("Comparing all models on the same test set...\n")
    results = compare_models()
    print_comparison_table(results)

    wine = load_wine()
    print("Confusion matrices:")
    for r in results:
        print(f"\nModel: {r['model']}")
        formatter_input = {
            "matrix": r["confusion_matrix"],
            "labels": wine.target_names.tolist(),
            "plot_path": "(not generated)",
            "classification_report": None,
        }
        print(format_confusion_matrix_result(formatter_input))


COMMAND_HANDLERS = {
    "train": handle_train,
    "compare": handle_compare,
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
