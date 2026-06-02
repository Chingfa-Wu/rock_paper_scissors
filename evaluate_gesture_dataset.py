"""Evaluate gesture recognition accuracy on a labeled image dataset."""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


VALID_CLASSES = ("rock", "paper", "scissors")
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
UNKNOWN_CLASS = "unknown"


def iter_labeled_images(dataset_dir: Path):
    for label in VALID_CLASSES:
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, image_path


def detect_image_gesture(cv2, detect_gesture, hands, image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        return UNKNOWN_CLASS

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)
    if not results.multi_hand_landmarks:
        return UNKNOWN_CLASS

    gesture = detect_gesture(results.multi_hand_landmarks[0])
    return gesture or UNKNOWN_CLASS


def empty_confusion_matrix(labels: list[str]) -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in labels} for actual in labels}


def build_per_class_results(
    matrix: dict[str, dict[str, int]],
) -> dict[str, dict[str, float | int | None]]:
    results = {}
    for label in VALID_CLASSES:
        total = sum(matrix[label].values())
        correct = matrix[label][label]
        results[label] = {
            "total": total,
            "correct": correct,
            "success_rate": correct / total if total else None,
        }
    return results


def evaluate_dataset(dataset_dir: Path) -> dict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    labeled_images = list(iter_labeled_images(dataset_dir))
    if not labeled_images:
        raise ValueError(
            "No test images found. Expected images under rock, paper, or scissors."
        )

    import cv2
    import mediapipe as mp

    from rock_paper_scissors import detect_gesture

    labels = [*VALID_CLASSES, UNKNOWN_CLASS]
    matrix = empty_confusion_matrix(labels)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        for actual, image_path in labeled_images:
            predicted = detect_image_gesture(cv2, detect_gesture, hands, image_path)
            matrix[actual][predicted] += 1

    per_class_results = build_per_class_results(matrix)
    total_images = sum(result["total"] for result in per_class_results.values())
    total_correct = sum(result["correct"] for result in per_class_results.values())
    per_class_accuracy = {
        label: result["success_rate"] for label, result in per_class_results.items()
    }

    return {
        "dataset": str(dataset_dir),
        "labels": labels,
        "total_images": total_images,
        "overall_accuracy": total_correct / total_images if total_images else None,
        "per_class_accuracy": per_class_accuracy,
        "per_class_results": per_class_results,
        "confusion_matrix": matrix,
        "most_confused": find_most_confused_pair(matrix),
    }


def find_most_confused_pair(matrix: dict[str, dict[str, int]]) -> dict | None:
    worst = None
    for actual in VALID_CLASSES:
        for predicted, count in matrix[actual].items():
            if predicted == actual or count == 0:
                continue
            if worst is None or count > worst["count"]:
                worst = {"actual": actual, "predicted": predicted, "count": count}
    return worst


def format_success_rate(rate: float | None) -> str:
    return "N/A" if rate is None else f"{rate * 100:.1f}%"


def render_confusion_matrix_svg(
    matrix: dict[str, dict[str, int]],
    labels: list[str],
    output_path: Path,
) -> None:
    rows = list(VALID_CLASSES)
    cell_size = 84
    left_margin = 116
    top_margin = 96
    right_margin = 128
    bottom_margin = 48
    width = left_margin + len(labels) * cell_size + right_margin
    height = top_margin + len(rows) * cell_size + bottom_margin
    max_count = max(
        [matrix[actual][predicted] for actual in rows for predicted in labels] or [0]
    )
    per_class_results = build_per_class_results(matrix)

    def color_for(count: int) -> str:
        if max_count == 0:
            return "rgb(244,248,252)"
        intensity = count / max_count
        red = round(232 - 126 * intensity)
        green = round(240 - 96 * intensity)
        blue = round(249 - 28 * intensity)
        return f"rgb({red},{green},{blue})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial, sans-serif" font-size="22" '
        'font-weight="700" fill="#172033">Gesture Confusion Matrix</text>',
        '<text x="24" y="58" font-family="Arial, sans-serif" font-size="13" '
        'fill="#526070">Rows are dataset labels; columns are model predictions.</text>',
    ]

    for column_index, label in enumerate(labels):
        x = left_margin + column_index * cell_size + cell_size / 2
        parts.append(
            f'<text x="{x}" y="{top_margin - 20}" font-family="Arial, sans-serif" '
            f'font-size="13" font-weight="700" text-anchor="middle" fill="#172033">'
            f"{html.escape(label)}</text>"
        )

    for row_index, actual in enumerate(rows):
        y = top_margin + row_index * cell_size
        text_y = y + cell_size / 2 + 5
        parts.append(
            f'<text x="{left_margin - 16}" y="{text_y}" font-family="Arial, sans-serif" '
            f'font-size="14" font-weight="700" text-anchor="end" fill="#172033">'
            f"{html.escape(actual)}</text>"
        )
        for column_index, predicted in enumerate(labels):
            x = left_margin + column_index * cell_size
            count = matrix[actual][predicted]
            fill = color_for(count)
            text_fill = "#ffffff" if max_count and count / max_count > 0.58 else "#172033"
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                    f'fill="{fill}" stroke="#d8e0ea"/>',
                    f'<text x="{x + cell_size / 2}" y="{text_y}" '
                    'font-family="Arial, sans-serif" font-size="22" '
                    f'font-weight="700" text-anchor="middle" fill="{text_fill}">{count}</text>',
                ]
            )

        rate = per_class_results[actual]["success_rate"]
        parts.append(
            f'<text x="{left_margin + len(labels) * cell_size + 24}" y="{text_y}" '
            'font-family="Arial, sans-serif" font-size="14" fill="#172033">'
            f'{format_success_rate(rate)}</text>'
        )

    parts.append(
        f'<text x="{left_margin + len(labels) * cell_size + 24}" y="{top_margin - 20}" '
        'font-family="Arial, sans-serif" font-size="13" font-weight="700" fill="#172033">'
        "Success</text>"
    )
    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate rock/paper/scissors recognition on labeled images."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset root with rock, paper, and scissors subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file path for saving the evaluation report.",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        help="Optional SVG file path for saving the confusion matrix chart.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = evaluate_dataset(args.dataset_dir)
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    output = json.dumps(report, ensure_ascii=False, indent=2)
    print(output)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")

    matrix_output = args.matrix_output
    if matrix_output is None and args.output:
        matrix_output = args.output.with_name(f"{args.output.stem}_confusion_matrix.svg")
    if matrix_output:
        render_confusion_matrix_svg(
            report["confusion_matrix"],
            report["labels"],
            matrix_output,
        )


if __name__ == "__main__":
    main()
