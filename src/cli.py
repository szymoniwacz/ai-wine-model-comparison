import sys
from src.trainer import train

AVAILABLE_COMMANDS = (
    "train",
    # Future: "experiment-max-depth", "experiment-confusion-matrix", "predict"
)


def print_usage():
    print("Usage:")
    print("  python -m src.cli train [--model MODEL_TYPE]")
    print("    --model MODEL_TYPE: decision_tree (default), logistic_regression, svm")
    # Future: add more commands here


def handle_train(args):
    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            model_type = args[idx + 1]
            print(f"Training model: {model_type}. Please wait...")
            result = train(model_type=model_type)
            print(f"Model trained. Accuracy: {result['accuracy']:.4f}")
            print(f"Model saved to: {result['model_path']}\n")
        else:
            print("Error: --model provided but no model type specified.")
    else:
        # Train all models if no specific model is provided
        all_models = ["decision_tree", "logistic_regression", "svm"]
        results = []
        print(
            "No --model specified. Training all models: decision_tree, logistic_regression, svm. Please wait...\n"
        )
        for model_type in all_models:
            result = train(model_type=model_type)
            results.append(result)
            print(
                f"Model: {model_type} | Accuracy: {result['accuracy']:.4f} | Saved to: {result['model_path']}"
            )
        print("\nAll models trained.")


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
