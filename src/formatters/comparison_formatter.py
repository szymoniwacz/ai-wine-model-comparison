"""
Formatter for model comparison CLI output (tables, highlights, colors).
"""


def print_comparison_table(results):
    bold = "\033[1m"
    green = "\033[92m"
    yellow = "\033[93m"
    cyan = "\033[96m"
    reset = "\033[0m"

    print(f"{bold}{cyan}Model comparison results:{reset}")
    header = f"{'Model':<20} {'Accuracy':<10} {'Model Path'}"
    print(f"{bold}{header}{reset}")
    print("=" * len(header))
    best_acc = max(r["accuracy"] for r in results)
    best_models = [r["model"] for r in results if r["accuracy"] == best_acc]
    for r in results:
        color = green if r["model"] in best_models else reset
        print(
            f"{color}{r['model']:<20} {r['accuracy']:<10.4f} {r['model_path']}{reset}"
        )
    print()
    print(
        f"{bold}Best model(s): {', '.join(best_models)} (accuracy: {best_acc:.4f}){reset}"
    )
    print()
    print(f"{bold}Accuracy differences:{reset}")
    for r in results:
        diff = best_acc - r["accuracy"]
        color = green if r["model"] in best_models else yellow if diff < 0.02 else reset
        print(f"{color}{r['model']:<20} {diff:+.4f}{reset}")
    print()


def print_model_reports(results, wine):
    bold = "\033[1m"
    blue = "\033[94m"
    reset = "\033[0m"
    from .confusion_matrix_formatter import format_confusion_matrix_result

    print(f"{bold}Confusion matrices and classification reports:{reset}")
    for idx, r in enumerate(results):
        print(f"\n{bold}{'-'*80}{reset}")
        print(f"{bold}Model: {r['model']}{reset}")
        formatter_input = {
            "matrix": r["confusion_matrix"],
            "labels": wine.target_names.tolist(),
            "plot_path": "(not generated)",
            "classification_report": r["classification_report"],
        }
        print(format_confusion_matrix_result(formatter_input))
        if idx == len(results) - 1:
            print(f"{bold}{'-'*80}{reset}")
