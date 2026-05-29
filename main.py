import cv2
import mediapipe as mp
import time

from config import *
from eye import calculate_ear
from yawn import calculate_mar
from head_pose import is_head_down

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

NOSE = 1
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263

cap = cv2.VideoCapture(0)

closed_start_time = None

def write_status(status):
    with open("status.txt", "w") as f:
        f.write(status)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NORMAL"

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            left_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in LEFT_EYE]
            right_eye = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in RIGHT_EYE]
            mouth = [(int(face.landmark[i].x*w), int(face.landmark[i].y*h)) for i in MOUTH]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            if ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()

                if time.time() - closed_start_time >= CLOSED_EYES_TIME:
                    status = "DROWSINESS ALERT!"
                else:
                    status = "EYES CLOSED"
            else:
                closed_start_time = None

            if mar > MAR_THRESHOLD:
                status = "YAWNING!"

            nose_y = face.landmark[NOSE].y
            eye_avg_y = (face.landmark[LEFT_EYE_CENTER].y +
                         face.landmark[RIGHT_EYE_CENTER].y) / 2

            if is_head_down(nose_y, eye_avg_y, HEAD_DOWN_THRESHOLD):
                status = "HEAD DOWN!"

        write_status(status)

        color = (0, 0, 255) if status != "NORMAL" else (0, 255, 0)

        cv2.putText(frame, f"Status: {status}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Driver Drowsiness System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()