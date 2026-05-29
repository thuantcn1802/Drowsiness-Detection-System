import cv2
import mediapipe as mp

# Khởi tạo MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Mở camera
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
) as face_detection:

    while True:
        success, frame = cap.read()

        if not success:
            print("Không đọc được camera")
            break

        # Lật ảnh cho giống gương
        frame = cv2.flip(frame, 1)

        # Chuyển sang RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face
        results = face_detection.process(rgb_frame)

        # Nếu có khuôn mặt
        if results.detections:
            for detection in results.detections:

                # Vẽ khung mặt
                mp_drawing.draw_detection(frame, detection)

        # Hiển thị
        cv2.imshow("Face Detection", frame)

        # Nhấn q để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()