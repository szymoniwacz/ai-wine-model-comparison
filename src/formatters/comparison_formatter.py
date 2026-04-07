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
    header = f"{'Rank':<5} {'Model':<20} {'Accuracy':<10} {'Δ vs best':<12} {'Model Path'}"
    print(f"{bold}{header}{reset}")
    print("=" * len(header))
    # Sort by accuracy descending
    sorted_results = sorted(results, key=lambda r: r['accuracy'], reverse=True)
    best_acc = sorted_results[0]["accuracy"]
    best_models = [r["model"] for r in sorted_results if r["accuracy"] == best_acc]
    for idx, r in enumerate(sorted_results, 1):
        diff = r["accuracy"] - best_acc
        percent = (diff / best_acc) * 100 if best_acc > 0 else 0
        color = green if r["model"] in best_models else yellow if abs(percent) < 2 else reset
        print(f"{color}{idx:<5} {r['model']:<20} {r['accuracy']:<10.4f} {percent:+8.2f}%   {r['model_path']}{reset}")
    print()
    print(f"{bold}Best model(s): {', '.join(best_models)} (accuracy: {best_acc:.4f}){reset}")
    if len(best_models) == 1:
        print(f"{green}The best model achieved the highest accuracy. Consider using it as your production baseline.{reset}")
    else:
        print(f"{yellow}Multiple models achieved the same best accuracy. You may choose based on speed, interpretability, or other factors.{reset}")
    print()
    print(f"{bold}Interpretation:{reset}")
    for r in sorted_results:
        if r["model"] in best_models:
            print(f"{green}- {r['model']} is a top performer on this dataset.{reset}")
        else:
            print(f"{yellow}- {r['model']} is {abs((r['accuracy']-best_acc)/best_acc)*100:.2f}% less accurate than the best model.{reset}")
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
