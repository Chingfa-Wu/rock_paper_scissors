"""Rock paper scissors game using MediaPipe landmarks and a trained KNN model."""

from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VALID_CLASSES = ("rock", "paper", "scissors")
MODELS_DIR = Path(__file__).with_name("models")
MODEL_PATH = MODELS_DIR / "gesture_knn.joblib"
HAND_LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
WRIST_INDEX = 0
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)
_gesture_model = None


def determine_winner(user_choice: str, pc_choice: str) -> str:
    """Return the game result for two valid rock/paper/scissors choices."""
    if user_choice == pc_choice:
        return "Tie!"

    win_conditions = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    return "You Win!" if win_conditions[user_choice] == pc_choice else "PC Wins!"


def create_hand_landmarker(
    model_path: Path = HAND_LANDMARKER_PATH,
    running_mode=vision.RunningMode.IMAGE,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Create a MediaPipe HandLandmarker for image or video landmark detection."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe hand landmarker model not found: {model_path}. "
            "Download hand_landmarker.task into the models folder."
        )

    base_options = python.BaseOptions(model_asset_buffer=model_path.read_bytes())
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def _mp_image_from_rgb(rgb_image):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)


def detect_image_hand_landmarks(landmarker, rgb_image):
    """Return the first detected hand landmarks from a still RGB image."""
    result = landmarker.detect(_mp_image_from_rgb(rgb_image))
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]


def detect_video_hand_landmarks(landmarker, rgb_frame, timestamp_ms: int):
    """Return the first detected hand landmarks from a video RGB frame."""
    result = landmarker.detect_for_video(_mp_image_from_rgb(rgb_frame), timestamp_ms)
    if not result.hand_landmarks:
        return None
    return result.hand_landmarks[0]


def _as_landmark_list(hand_landmarks):
    return getattr(hand_landmarks, "landmark", hand_landmarks)


def extract_landmark_features(hand_landmarks) -> list[float] | None:
    """Convert 21 MediaPipe hand landmarks into normalized KNN features."""
    landmarks = _as_landmark_list(hand_landmarks)
    if len(landmarks) != 21:
        return None

    wrist = landmarks[WRIST_INDEX]
    wrist_z = getattr(wrist, "z", 0.0)

    relative_points = []
    max_distance = 0.0
    for landmark in landmarks:
        dx = landmark.x - wrist.x
        dy = landmark.y - wrist.y
        dz = getattr(landmark, "z", 0.0) - wrist_z
        relative_points.append((dx, dy, dz))
        max_distance = max(max_distance, math.sqrt(dx * dx + dy * dy + dz * dz))

    if max_distance == 0:
        return None

    features = []
    for dx, dy, dz in relative_points:
        features.extend((dx / max_distance, dy / max_distance, dz / max_distance))
    return features


def load_gesture_model(model_path: Path = MODEL_PATH):
    """Load the trained KNN gesture model."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"KNN model not found: {model_path}. "
            "Run `python train_gesture_knn.py` first."
        )

    import joblib

    return joblib.load(model_path)


def get_gesture_model():
    """Load once and reuse the KNN model while the camera loop is running."""
    global _gesture_model
    if _gesture_model is None:
        _gesture_model = load_gesture_model()
    return _gesture_model


def detect_gesture(hand_landmarks, model=None) -> str | None:
    """Predict rock, paper, or scissors from MediaPipe landmarks with KNN."""
    features = extract_landmark_features(hand_landmarks)
    if features is None:
        return None

    classifier = model or get_gesture_model()
    prediction = classifier.predict([features])[0]
    if prediction in VALID_CLASSES:
        return str(prediction)
    return None


def draw_hand_landmarks(frame, hand_landmarks) -> None:
    """Draw MediaPipe landmarks on the OpenCV frame."""
    landmarks = _as_landmark_list(hand_landmarks)
    height, width = frame.shape[:2]

    points = []
    for landmark in landmarks:
        x = min(max(int(landmark.x * width), 0), width - 1)
        y = min(max(int(landmark.y * height), 0), height - 1)
        points.append((x, y))

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 200, 255), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def main() -> None:
    """Run the webcam rock paper scissors game."""
    try:
        gesture_model = load_gesture_model()
        hand_landmarker = create_hand_landmarker(
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.7,
            min_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        print(f"[Error] {exc}")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        hand_landmarker.close()
        print("[Error] Could not open the camera.")
        return

    result_text = "Press SPACE to play!"
    pc_choice_text = ""
    frame_index = 0

    print("[Info] Press SPACE to play. Press 'q' to quit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[Error] Could not read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_index += 1
            hand_landmarks = detect_video_hand_landmarks(
                hand_landmarker,
                rgb_frame,
                frame_index * 33,
            )
            current_gesture = None

            if hand_landmarks:
                draw_hand_landmarks(frame, hand_landmarks)
                current_gesture = detect_gesture(hand_landmarks, gesture_model)
                if current_gesture:
                    cv2.putText(
                        frame,
                        f"Detected: {current_gesture}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        3,
                    )

            cv2.putText(
                frame,
                result_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            if pc_choice_text:
                cv2.putText(
                    frame,
                    pc_choice_text,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3,
                )

            cv2.imshow("Rock Paper Scissors - MediaPipe KNN", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if current_gesture:
                    pc_choice = random.choice(VALID_CLASSES)
                    result = determine_winner(current_gesture, pc_choice)
                    result_text = f"Result: {result} (You: {current_gesture})"
                    pc_choice_text = f"PC Choice: {pc_choice}"
                else:
                    result_text = "Please show a clear gesture!"
                    pc_choice_text = ""
    finally:
        camera.release()
        hand_landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
