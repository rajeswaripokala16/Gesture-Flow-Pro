"""
Advanced Real-Time Hand Gesture Detection System (No keyboard dependency)
"""

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import threading
from collections import deque

# Disable PyAutoGUI failsafe for demo (ESC exits via cv2)
pyautogui.FAILSAFE = False

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

class GestureDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Gesture state tracking
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = 0
        self.gesture_threshold = 5
        
        # Gesture mapping
        self.gestures = {
            'thumbs_up': self.thumbs_up_action,
            'peace': self.peace_action,
            'fist': self.fist_action,
            'open_palm': self.open_palm_action,
            'pointing': self.pointing_action
        }
        
        # Control states
        self.volume_level = 50
        self.is_presenting = False
        
        print("=== Hand Gesture Control System ===")
        print("Controls: ESC = Exit")
        print("Gestures: Thumbs Up (Next), Peace (Vol+), Fist (Vol-), Palm (Present), Point (Mouse)")
    
    def detect_gesture(self, landmarks):
        """Classify gesture from hand landmarks"""
        if len(landmarks) != 21:
            return None
            
        # Key landmarks (MediaPipe topology)
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Finger states (tip above PIP = extended)
        thumb_extended = thumb_tip[1] < landmarks[3][1]
        index_extended = index_tip[2] < landmarks[6][2]
        middle_extended = middle_tip[2] < landmarks[10][2]
        ring_extended = ring_tip[2] < landmarks[14][2]
        pinky_extended = pinky_tip[2] < landmarks[18][2]
        
        # Gesture classification
        if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'thumbs_up'
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            return 'peace'
        elif not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'fist'
        elif index_extended and middle_extended and ring_extended and pinky_extended:
            return 'open_palm'
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return 'pointing'
        
        return None
    
    def thumbs_up_action(self):
        if self.is_presenting:
            pyautogui.hotkey('right')  # Next slide
        else:
            pyautogui.press('space')  # Play/pause
        print("👆 Thumbs Up - Next/Play")
    
    def peace_action(self):
        pyautogui.press('volumeup')
        print("✌️ Peace - Volume Up")
    
    def fist_action(self):
        pyautogui.press('volumedown')
        print("✊ Fist - Volume Down")
    
    def open_palm_action(self):
        self.is_presenting = not self.is_presenting
        status = "ON" if self.is_presenting else "OFF"
        print(f"🖐️ Palm - Presentation {status}")
    
    def pointing_action(self):
        print("👆 Pointing - Mouse Active")
    
    def execute_gesture(self, gesture):
        current_time = time.time()
        if current_time - self.last_gesture_time > 0.5:
            if gesture in self.gestures:
                self.gestures[gesture]()
            self.last_gesture_time = current_time
    
    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                
                landmarks = np.array([[lm.x, lm.y, lm.z] 
                                    for lm in hand_landmarks.landmark])
                
                gesture = self.detect_gesture(landmarks)
                if gesture:
                    self.gesture_history.append(gesture)
                    
                    if len(self.gesture_history) >= self.gesture_threshold:
                        confirmed = max(set(self.gesture_history), 
                                      key=self.gesture_history.count)
                        self.execute_gesture(confirmed)
                        self.gesture_history.clear()
        
        return frame
    
    def run(self):
        print("Starting... Press ESC to exit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            
            # Status overlay
            status = f"Gestures: {len(self.gesture_history)} | Mode: {'PRESENT' if self.is_presenting else 'MEDIA'}"
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            cv2.imshow("Hand Gesture Control", frame)
            
            # ESC = 27 to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = GestureDetector()
    detector.run()
