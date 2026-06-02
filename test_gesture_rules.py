"""Tests for gesture rules and evaluation metrics."""

import sys
import types
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


fake_hands = SimpleNamespace(HandLandmark=FakeHandLandmark)
fake_mediapipe = SimpleNamespace(
    solutions=SimpleNamespace(
        hands=fake_hands,
        drawing_utils=SimpleNamespace(),
    )
)
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("mediapipe", fake_mediapipe)

from evaluate_gesture_dataset import (
    build_per_class_results,
    empty_confusion_matrix,
    find_most_confused_pair,
    render_confusion_matrix_svg,
)
from rock_paper_scissors import VALID_CLASSES, detect_gesture


mp_hands = fake_hands


def fake_landmarks(open_fingers: set[int]):
    landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
    wrist = mp_hands.HandLandmark.WRIST
    landmarks[wrist] = SimpleNamespace(x=0.0, y=0.0)

    pairs = (
        (mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP),
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP),
    )
    for finger_index, (tip, pip) in enumerate(pairs):
        landmarks[pip] = SimpleNamespace(x=1.0, y=0.0)
        landmarks[tip] = SimpleNamespace(
            x=2.0 if finger_index in open_fingers else 0.5,
            y=0.0,
        )
    return SimpleNamespace(landmark=landmarks)


class GestureRuleTests(unittest.TestCase):
    def test_detects_all_expected_classes(self):
        cases = {
            "rock": set(),
            "scissors": {1, 2},
            "paper": {1, 2, 3, 4},
        }
        for expected, open_fingers in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(detect_gesture(fake_landmarks(open_fingers)), expected)

    def test_three_open_fingers_is_not_paper(self):
        self.assertIsNone(detect_gesture(fake_landmarks({1, 2, 3})))

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
