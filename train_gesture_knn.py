"""Train and evaluate a KNN gesture classifier from a labeled image dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rock_paper_scissors import (
    MODEL_PATH,
    VALID_CLASSES,
    create_hand_landmarker,
    detect_image_hand_landmarks,
    extract_landmark_features,
)


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def iter_labeled_images(dataset_dir: Path):
    """Yield images from dataset_dir/rock, dataset_dir/paper, dataset_dir/scissors."""
    for label in VALID_CLASSES:
        label_dir = dataset_dir / label
        if not label_dir.exists():
            continue
        for image_path in sorted(label_dir.iterdir()):
            if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield label, image_path


def extract_image_feature(cv2_module, hands, image_path: Path) -> list[float] | None:
    image = cv2_module.imread(str(image_path))
    if image is None:
        return None

    rgb_image = cv2_module.cvtColor(image, cv2_module.COLOR_BGR2RGB)
    hand_landmarks = detect_image_hand_landmarks(hands, rgb_image)
    if not hand_landmarks:
        return None

    return extract_landmark_features(hand_landmarks)


def build_feature_dataset(
    dataset_dir: Path,
    min_detection_confidence: float,
) -> tuple[list[list[float]], list[str], dict[str, int]]:
    images = list(iter_labeled_images(dataset_dir))
    if not images:
        raise ValueError(
            "No training images found. Expected files under rock, paper, or scissors."
        )

    features = []
    labels = []
    skipped = {"unreadable_or_no_hand": 0}
    landmarker = create_hand_landmarker(
        min_detection_confidence=min_detection_confidence,
    )
    try:
        for label, image_path in images:
            feature = extract_image_feature(cv2, landmarker, image_path)
            if feature is None:
                skipped["unreadable_or_no_hand"] += 1
                continue
            features.append(feature)
            labels.append(label)
    finally:
        landmarker.close()

    if not features:
        raise ValueError("MediaPipe did not detect a usable hand in any image.")

    return features, labels, skipped


def train_knn_model(
    features: list[list[float]],
    labels: list[str],
    n_neighbors: int,
):
    model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance"),
    )
    model.fit(features, labels)
    return model


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def train_and_evaluate(args: argparse.Namespace) -> dict:
    features, labels, skipped = build_feature_dataset(
        args.dataset_dir,
        args.min_detection_confidence,
    )
    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=labels,
    )

    n_neighbors = min(args.neighbors, len(train_x))
    model = train_knn_model(train_x, train_y, n_neighbors)
    predictions = model.predict(test_x)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)

    matrix = confusion_matrix(test_y, predictions, labels=list(VALID_CLASSES))
    report = {
        "dataset": display_path(args.dataset_dir),
        "model_output": display_path(args.output),
        "total_features": len(features),
        "train_count": len(train_x),
        "test_count": len(test_x),
        "skipped": skipped,
        "n_neighbors": n_neighbors,
        "accuracy": accuracy_score(test_y, predictions),
        "labels": list(VALID_CLASSES),
        "classification_report": classification_report(
            test_y,
            predictions,
            labels=list(VALID_CLASSES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": matrix.tolist(),
    }

    if args.report_output:
        args.report_output.parent.mkdir(parents=True, exist_ok=True)
        args.report_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a sklearn KNN model from MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help="Dataset root with rock, paper, and scissors subfolders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_PATH,
        help="Where to save the trained KNN model.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of detected samples reserved for testing.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="K value for KNN.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split.",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand detection confidence for dataset extraction.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        report = train_and_evaluate(parse_args())
    except (FileNotFoundError, ModuleNotFoundError, ValueError) as exc:
        print(f"[Error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
