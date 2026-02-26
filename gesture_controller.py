
import cv2
import mediapipe as mp
import pyautogui
import math
import time
import threading
import logging

class GestureController:
    def __init__(self, settings):
        self.settings = settings
        self.running = False
        self.thread = None

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False

    def run(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            logging.error("Camera not accessible")
            return

        screen_w, screen_h = pyautogui.size()
        last_click = 0

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    index = hand_landmarks.landmark[8]
                    x = int(index.x * screen_w)
                    y = int(index.y * screen_h)
                    pyautogui.moveTo(x, y)

                    thumb = hand_landmarks.landmark[4]
                    dist = math.dist([thumb.x, thumb.y], [index.x, index.y]) * frame.shape[1]

                    if dist < self.settings["gesture_sensitivity"]:
                        if time.time() - last_click > self.settings["cooldown"]:
                            pyautogui.click()
                            last_click = time.time()

            cv2.imshow("Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
