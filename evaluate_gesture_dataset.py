"""Evaluate gesture recognition accuracy on a labeled image dataset."""

from __future__ import annotations

import argparse
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
    totals = {label: 0 for label in VALID_CLASSES}
    correct = {label: 0 for label in VALID_CLASSES}

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        for actual, image_path in labeled_images:
            predicted = detect_image_gesture(cv2, detect_gesture, hands, image_path)
            totals[actual] += 1
            correct[actual] += int(predicted == actual)
            matrix[actual][predicted] += 1

    total_images = sum(totals.values())
    total_correct = sum(correct.values())
    per_class_accuracy = {
        label: (correct[label] / totals[label] if totals[label] else None)
        for label in VALID_CLASSES
    }

    return {
        "dataset": str(dataset_dir),
        "labels": labels,
        "total_images": total_images,
        "overall_accuracy": total_correct / total_images if total_images else None,
        "per_class_accuracy": per_class_accuracy,
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


if __name__ == "__main__":
    main()
