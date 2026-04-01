"""
Real-time FaceMesh: FIXED VERSION
No more AttributeError or ValueError
"""

import cv2
import mediapipe as mp
import numpy as np

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# FIXED: Correct plural constants
FACEMESH_CONTOURS = mp_face_mesh.FACEMESH_CONTOURS
FACEMESH_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
FACEMESH_IRISES = mp_face_mesh.FACEMESH_IRISES

def draw_landmarks(image, detection):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=detection.multi_face_landmarks[0],
        connections=FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
    )
    
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=detection.multi_face_landmarks[0],
        connections=FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
    )
    
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=detection.multi_face_landmarks[0],
        connections=FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1)
    )
    
    return image

def calculate_angles(frame_shape, landmarks):
    """FIXED: Use frame shape, not landmarks shape"""
    h, w = frame_shape[:2]
    
    # Key landmarks indices
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    NOSE_TIP = 1
    LEFT_MOUTH_CORNER = 61
    RIGHT_MOUTH_CORNER = 291
    CHIN = 152
    
    # Convert normalized landmarks to pixel coordinates
    left_eye_top = landmarks[LEFT_EYE_TOP]
    left_eye_bottom = landmarks[LEFT_EYE_BOTTOM]
    right_eye_top = landmarks[RIGHT_EYE_TOP]
    right_eye_bottom = landmarks[RIGHT_EYE_BOTTOM]
    nose_tip = landmarks[NOSE_TIP]
    left_mouth = landmarks[LEFT_MOUTH_CORNER]
    right_mouth = landmarks[RIGHT_MOUTH_CORNER]
    chin = landmarks[CHIN]
    
    # Eye aspect ratio
    left_eye_h = left_eye_top.y - left_eye_bottom.y
    right_eye_h = right_eye_top.y - right_eye_bottom.y
    ear = (left_eye_h + right_eye_h) / 2.0
    
    # Mouth openness
    mouth_h = right_mouth.y - nose_tip.y
    
    # Head tilt (simplified)
    nose_x = nose_tip.x * w
    chin_x = chin.x * w
    head_tilt = (nose_x - chin_x) / w * 180
    
    return {
        'ear': ear,
        'mouth': mouth_h,
        'tilt': head_tilt
    }

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("FaceMesh FIXED - Press 'q' to quit")
    
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                # Draw landmarks
                frame = draw_landmarks(frame, results)
                
                # FIXED: Pass frame.shape and raw landmarks
                metrics = calculate_angles(
                    frame.shape, 
                    results.multi_face_landmarks[0].landmark
                )
                
                # Overlay metrics
                cv2.putText(frame, f"EAR: {metrics['ear']:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Mouth: {metrics['mouth']:.1f}px", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                cv2.putText(frame, f"Tilt: {metrics['tilt']:.0f}°", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.putText(frame, "FaceMesh - 468 Landmarks | Press 'q'", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            
            cv2.imshow("FaceMesh - FIXED VERSION", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
