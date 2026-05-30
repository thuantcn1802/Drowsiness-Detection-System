import cv2
import mediapipe as mp
import time

from src.config.config import *
from src.detection.eye import calculate_ear
from src.detection.yawn import calculate_mar
from src.detection.head_pose import HeadDownDetector

try:
    from cnn_predictor import CNNPredictor
except Exception as exc:
    CNNPredictor = None
    print(f"Khong import duoc CNNPredictor: {exc}")


mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

cap = cv2.VideoCapture(CAMERA_INDEX)

closed_start_time = None
frame_count = 0
cnn_result = None

head_detector = HeadDownDetector(
    calibration_frames=HEAD_CALIBRATION_FRAMES,
    score_delta=HEAD_DOWN_SCORE_DELTA,
    down_time=HEAD_DOWN_TIME,
)

cnn_predictor = None
if ENABLE_CNN_MODEL and CNNPredictor is not None:
    try:
        cnn_predictor = CNNPredictor()
        print(f"Da load CNN model: {CNN_MODEL_PATH}")
    except Exception as exc:
        cnn_predictor = None
        print(f"Khong load duoc CNN model, he thong se chay rule-based: {exc}")


def write_status(status):
    with open("status.txt", "w", encoding="utf-8") as f:
        f.write(status)


def get_cnn_text(result):
    if result is None:
        return "CNN: OFF"

    label = result["label"]
    conf = result["confidence"]
    count = result["drowsy_count"]
    size = result["history_size"]

    return f"CNN: {label} {conf:.2f} ({count}/{size})"


with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NORMAL"

        ear = 0.0
        mar = 0.0
        head_score = 0.0
        head_delta = 0.0
        is_calibrated = head_detector.is_calibrated

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            left_eye = [
                (int(face.landmark[i].x * w), int(face.landmark[i].y * h))
                for i in LEFT_EYE
            ]

            right_eye = [
                (int(face.landmark[i].x * w), int(face.landmark[i].y * h))
                for i in RIGHT_EYE
            ]

            mouth = [
                (int(face.landmark[i].x * w), int(face.landmark[i].y * h))
                for i in MOUTH
            ]

            ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
            mar = calculate_mar(mouth)

            # 1. Phat hien nham mat lau bang EAR
            if ear < EAR_THRESHOLD:
                if closed_start_time is None:
                    closed_start_time = time.time()

                if time.time() - closed_start_time >= CLOSED_EYES_TIME:
                    status = "DROWSINESS ALERT!"
                else:
                    status = "EYES CLOSED"
            else:
                closed_start_time = None

            # 2. Phat hien ngap bang MAR
            if mar > MAR_THRESHOLD:
                status = "YAWNING!"

            # 3. Phat hien guc/cui dau bang HeadDownDetector
            is_head_alert, head_score, head_delta, is_calibrated = head_detector.update(face)

            if is_head_alert:
                status = "HEAD DOWN!"

            # 4. Du doan bang CNN/FastKAN
            if cnn_predictor is not None:
                try:
                    if frame_count % CNN_PREDICT_EVERY_N_FRAMES == 0:
                        cnn_result = cnn_predictor.predict(frame, face)

                    if cnn_result is not None and cnn_result["alert"]:
                        status = "DROWSINESS ALERT!"
                except Exception as exc:
                    print(f"Loi khi CNN predict: {exc}")
                    cnn_result = None

        else:
            status = "NO FACE"
            closed_start_time = None
            head_detector.reset_timer()

        write_status(status)

        color = (
            (0, 0, 255)
            if status not in ["NORMAL", "EYES CLOSED", "NO FACE"]
            else (0, 255, 0)
        )

        cv2.putText(frame, f"Status: {status}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, f"Head score: {head_score:.3f}", (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.putText(frame, f"Head delta: {head_delta:.3f}", (30, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.putText(frame, get_cnn_text(cnn_result), (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)

        if not is_calibrated:
            cv2.putText(frame,
                        f"Calibrating head: {len(head_detector.neutral_scores)}/{HEAD_CALIBRATION_FRAMES}",
                        (30, 255),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 255, 255),
                        2)

        cv2.putText(frame, "Press R to reset head calibration", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        cv2.imshow("Driver Drowsiness System", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("r"):
            head_detector.reset()
            if cnn_predictor is not None:
                cnn_predictor.history.clear()
            cnn_result = None
            print("Da reset calibration guc dau va CNN history")


cap.release()
cv2.destroyAllWindows()