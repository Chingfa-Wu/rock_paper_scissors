"""使用攝影機與 MediaPipe 進行即時剪刀石頭布遊戲。"""

import math
import random

import cv2
import mediapipe as mp


VALID_CLASSES = ("rock", "paper", "scissors")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def determine_winner(user_choice: str, pc_choice: str) -> str:
    """判斷使用者與電腦出拳的勝負。"""
    if user_choice == pc_choice:
        return "Tie!"

    win_conditions = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    return "You Win!" if win_conditions[user_choice] == pc_choice else "PC Wins!"


def get_distance(p1, p2) -> float:
    """計算兩個 MediaPipe 節點之間的歐幾里得距離。"""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def detect_gesture(hand_landmarks) -> str | None:
    """根據四指是否伸直，辨識石頭、剪刀或布。"""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    finger_tips = (
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    )
    finger_pips = (
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    )

    finger_status = []
    for tip_index, pip_index in zip(finger_tips, finger_pips):
        tip = hand_landmarks.landmark[tip_index]
        pip = hand_landmarks.landmark[pip_index]
        finger_status.append(int(get_distance(tip, wrist) > get_distance(pip, wrist)))

    # 大拇指在鏡頭中的方向變化較大，僅以其餘四指進行分類。
    open_finger_count = sum(finger_status[1:])
    if open_finger_count == 0:
        return "rock"
    if (
        open_finger_count == 2
        and finger_status[1] == 1
        and finger_status[2] == 1
    ):
        return "scissors"
    if open_finger_count >= 3:
        return "paper"
    return None


def main() -> None:
    """啟動攝影機、偵測手勢，並執行遊戲迴圈。"""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[Error] 無法開啟攝影機，請確認攝影機權限與裝置狀態。")
        return

    result_text = "Press SPACE to play!"
    pc_choice_text = ""

    print("[Info] 骨架辨識啟動！按下空白鍵出拳，按下 'q' 離開。")
    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        ) as hands:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("[Error] 無法讀取攝影機畫面。")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                current_gesture = None

                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )
                    current_gesture = detect_gesture(hand_landmarks)
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

                cv2.imshow("Rock Paper Scissors - MediaPipe", frame)
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
