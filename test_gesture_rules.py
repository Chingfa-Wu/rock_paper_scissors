"""Tests for KNN gesture plumbing and evaluation metrics."""

from pathlib import Path
from types import SimpleNamespace
import unittest


class FakeHandLandmark:
    WRIST = 0
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    THUMB_IP = 3
    INDEX_FINGER_PIP = 6
    MIDDLE_FINGER_PIP = 10
    RING_FINGER_PIP = 14
    PINKY_PIP = 18


from evaluate_gesture_dataset import (
    build_per_class_results,
    empty_confusion_matrix,
    find_most_confused_pair,
    render_confusion_matrix_svg,
)
from rock_paper_scissors import (
    VALID_CLASSES,
    detect_gesture,
    extract_landmark_features,
)


mp_hands = SimpleNamespace(HandLandmark=FakeHandLandmark)


class FakeModel:
    def __init__(self, prediction: str):
        self.prediction = prediction
        self.seen_features = None

    def predict(self, rows):
        self.seen_features = rows[0]
        return [self.prediction]


def fake_landmarks(open_fingers: set[int]):
    landmarks = [SimpleNamespace(x=0.0, y=0.0, z=0.0) for _ in range(21)]
    wrist = mp_hands.HandLandmark.WRIST
    landmarks[wrist] = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    pairs = (
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    )
    for finger_index, (tip, pip) in enumerate(pairs):
        landmarks[pip] = SimpleNamespace(x=1.0, y=0.0, z=0.0)
        landmarks[tip] = SimpleNamespace(
            x=2.0 if finger_index in open_fingers else 0.5,
            y=0.0,
            z=0.0,
        )
    return SimpleNamespace(landmark=landmarks)


class GestureKnnTests(unittest.TestCase):
    def test_landmarks_become_normalized_knn_features(self):
        features = extract_landmark_features(fake_landmarks({1, 2}))

        self.assertEqual(len(features), 63)
        self.assertEqual(features[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(max(abs(value) for value in features), 1.0)

    def test_detect_gesture_uses_model_prediction(self):
        model = FakeModel("paper")

        self.assertEqual(detect_gesture(fake_landmarks(set()), model), "paper")
        self.assertEqual(len(model.seen_features), 63)

    def test_detect_gesture_rejects_unknown_model_output(self):
        self.assertIsNone(detect_gesture(fake_landmarks({1, 2, 3}), FakeModel("lizard")))

    def test_confusion_matrix_reports_worst_mistake(self):
        labels = [*VALID_CLASSES, "unknown"]
        matrix = empty_confusion_matrix(labels)
        matrix["rock"]["paper"] = 2
        matrix["scissors"]["unknown"] = 1

        self.assertEqual(
            find_most_confused_pair(matrix),
            {"actual": "rock", "predicted": "paper", "count": 2},
        )

    def test_per_class_results_include_success_rates(self):
        labels = [*VALID_CLASSES, "unknown"]
        matrix = empty_confusion_matrix(labels)
        matrix["rock"]["rock"] = 2
        matrix["rock"]["paper"] = 1
        matrix["paper"]["paper"] = 4

        self.assertEqual(
            build_per_class_results(matrix),
            {
                "rock": {"total": 3, "correct": 2, "success_rate": 2 / 3},
                "paper": {"total": 4, "correct": 4, "success_rate": 1.0},
                "scissors": {"total": 0, "correct": 0, "success_rate": None},
            },
        )

    def test_confusion_matrix_svg_contains_counts_and_success_rates(self):
        labels = [*VALID_CLASSES, "unknown"]
        matrix = empty_confusion_matrix(labels)
        matrix["rock"]["rock"] = 1
        matrix["rock"]["unknown"] = 1
        output_path = Path(__file__).with_name("test_confusion_matrix.svg")

        try:
            render_confusion_matrix_svg(matrix, labels, output_path)
            svg = output_path.read_text(encoding="utf-8")
        finally:
            if output_path.exists():
                output_path.unlink()

        self.assertIn("Gesture Confusion Matrix", svg)
        self.assertIn(">50.0%</text>", svg)
        self.assertIn(">unknown</text>", svg)


if __name__ == "__main__":
    unittest.main()
