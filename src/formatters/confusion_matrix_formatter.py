def format_confusion_matrix_result(result):
    matrix = result["matrix"]
    labels = result["labels"]
    plot_path = result["plot_path"]

    lines = []

    lines.append("")
    lines.append("=== Confusion Matrix ===")

    first_col_width = 20
    col_width = 14

    header = "actual \\ predicted".ljust(first_col_width)
    for label in labels:
        header += f"{label:>{col_width}}"
    lines.append(header)
    lines.append("-" * (first_col_width + col_width * len(labels)))

    for index, row in enumerate(matrix):
        line = f"{labels[index]:<{first_col_width}}"
        for value in row:
            line += f"{value:>{col_width}}"
        lines.append(line)

    total_predictions = sum(sum(row) for row in matrix)
    correct_predictions = sum(matrix[i][i] for i in range(len(matrix)))
    mistakes = total_predictions - correct_predictions

    lines.append("")
    lines.append("Summary:")
    lines.append(f"- Correct predictions: {correct_predictions}/{total_predictions}")
    lines.append(f"- Mistakes: {mistakes}")

    confusion_found = False
    for actual_index, row in enumerate(matrix):
        for predicted_index, value in enumerate(row):
            if actual_index != predicted_index and value > 0:
                lines.append(
                    f"- {value} sample(s) of '{labels[actual_index]}' "
                    f"were predicted as '{labels[predicted_index]}'"
                )
                confusion_found = True

    if not confusion_found:
        lines.append("- No class confusions detected")

    lines.append("")
    lines.append("Classification report:")
    lines.extend(result["classification_report"].splitlines())

    lines.append("")
    return "\n".join(lines)
