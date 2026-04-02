import sys
from sklearn.datasets import load_wine
from src.trainer import train
from src.formatters.confusion_matrix_formatter import format_confusion_matrix_result

AVAILABLE_COMMANDS = (
    "train",
    # Future: "experiment-max-depth", "experiment-confusion-matrix", "predict"
)


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
    all_models = ["decision_tree", "logistic_regression", "svm"]
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
            "No --model specified. Training all models: decision_tree, logistic_regression, svm. Please wait..."
        )
        for model_type in all_models:
            print(f"-" * 120)
            print(f"Training model: {model_type}. Please wait...\n")
            result = train(model_type=model_type)
            print_model_result(model_type, result)
            print(f"-" * 120) if model_type == all_models[-1] else ""
        print("All models trained.\n")


COMMAND_HANDLERS = {
    "train": handle_train,
    # Future: add more handlers here
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
        print(f"Available commands: {', '.join(AVAILABLE_COMMANDS)}")
        print_usage()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
