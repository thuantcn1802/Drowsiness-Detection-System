import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25
CLOSED_EYES_TIME = 2  # nhắm mắt quá 2 giây thì cảnh báo

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(eye_points):
    p1, p2, p3, p4, p5, p6 = eye_points

    vertical1 = euclidean_distance(p2, p6)
    vertical2 = euclidean_distance(p3, p5)
    horizontal = euclidean_distance(p1, p4)

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

cap = cv2.VideoCapture(0)

closed_start_time = None
is_alerting = False

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()

        if not success:
            print("Không đọc được camera")
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status = "NORMAL"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            left_eye_points = []
            right_eye_points = []

            for idx in LEFT_EYE:
                lm = face_landmarks.landmark[idx]
                left_eye_points.append((int(lm.x * w), int(lm.y * h)))

            for idx in RIGHT_EYE:
                lm = face_landmarks.landmark[idx]
                right_eye_points.append((int(lm.x * w), int(lm.y * h)))

            left_ear = calculate_ear(left_eye_points)
            right_ear = calculate_ear(right_eye_points)
            avg_ear = (left_ear + right_ear) / 2.0

            for point in left_eye_points + right_eye_points:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)

            if avg_ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()

                closed_duration = time.time() - closed_start_time

                if closed_duration >= CLOSED_EYES_TIME:
                    status = "DROWSINESS ALERT!"
                    is_alerting = True
                else:
                    status = "EYES CLOSED"
            else:
                closed_start_time = None
                is_alerting = False
                status = "NORMAL"

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        color = (0, 0, 255) if is_alerting else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Drowsiness EAR Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()