import customtkinter as ctk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import time
import pygame
import os
import shutil

from src.config.config import *
from src.detection.eye import calculate_ear
from src.detection.yawn import calculate_mar
from src.detection.head_pose import HeadDownDetector

try:
    from src.detection.cnn_predictor import CNNPredictor
except Exception as exc:
    CNNPredictor = None
    print(f"Khong import duoc CNNPredictor: {exc}")


# Fallback config nếu config.py chưa có biến mới
try:
    AI_YAWNING_CONF_THRESHOLD
except NameError:
    AI_YAWNING_CONF_THRESHOLD = 0.85

try:
    AI_YAWNING_MAR_THRESHOLD
except NameError:
    AI_YAWNING_MAR_THRESHOLD = 0.50

try:
    APP_WIDTH
except NameError:
    APP_WIDTH = 1500
    APP_HEIGHT = 850
    CAMERA_DISPLAY_WIDTH = 1180
    CAMERA_DISPLAY_HEIGHT = 665

try:
    CNN_TEXT_SCALE
except NameError:
    CNN_TEXT_SCALE = 0.65
    CNN_TEXT_THICKNESS = 2
    CNN_PROB_TEXT_SCALE = 0.45
    CNN_PROB_TEXT_THICKNESS = 1
    CNN_TEXT_X = 30
    CNN_TEXT_Y = 45
    CNN_PROB_START_Y = 75
    CNN_PROB_LINE_GAP = 22


# Cấu hình giao diện
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

pygame.mixer.init()
alarm_playing = False

app = ctk.CTk()
app.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
app.title("AI Driver Dashboard")


# Biến toàn cục
is_running = False
closed_start_time = None
trip_start_time = None
blink_flag = False

drowsy_count = 0
yawn_count = 0
head_count = 0
ai_count = 0
last_status = "NORMAL"

frame_count = 0
cnn_result = None

# Âm thanh cảnh báo
def play_alarm():
    global alarm_playing

    if not alarm_playing:
        pygame.mixer.music.play(-1)
        alarm_playing = True


def stop_alarm():
    global alarm_playing

    if alarm_playing:
        pygame.mixer.music.stop()
        alarm_playing = False


# Nút điều khiển camera
def start_camera():
    global is_running, trip_start_time, closed_start_time
    global frame_count, cnn_result

    is_running = True
    closed_start_time = None
    frame_count = 0
    cnn_result = None

    head_detector.reset()

    if cnn_predictor is not None:
        cnn_predictor.history.clear()

    if trip_start_time is None:
        trip_start_time = time.time()


def stop_camera():
    global is_running

    is_running = False
    stop_alarm()
    status_big.configure(text="STOPPED", text_color="#ff4d4d")

def clear_debug_images():
    """Xoa anh debug CNN cua phien truoc truoc khi chuong trinh bat dau."""
    try:
        if os.path.exists(CNN_DEBUG_DIR):
            for filename in os.listdir(CNN_DEBUG_DIR):
                file_path = os.path.join(CNN_DEBUG_DIR, filename)

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

            print(f"Da xoa anh debug cu trong thu muc: {CNN_DEBUG_DIR}")
        else:
            os.makedirs(CNN_DEBUG_DIR, exist_ok=True)
            print(f"Da tao thu muc debug: {CNN_DEBUG_DIR}")

    except Exception as exc:
        print(f"Khong xoa duoc anh debug cu: {exc}")

clear_debug_images()

def exit_app():
    stop_alarm()

    pygame.mixer.quit()
    cap.release()
    face_mesh.close()
    app.destroy()

# Layout
LEFT_PANEL_WIDTH = 300
CARD_RADIUS = 14

left = ctk.CTkFrame(app, width=LEFT_PANEL_WIDTH, fg_color="#0b1324", corner_radius=0)
left.pack(side="left", fill="y")
left.pack_propagate(False)

right = ctk.CTkFrame(app, fg_color="#0f172a")
right.pack(side="right", expand=True, fill="both")

sidebar = ctk.CTkFrame(left, fg_color="transparent")
sidebar.pack(fill="both", expand=True, padx=14, pady=12)

# Header
brand_card = ctk.CTkFrame(
    sidebar,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
brand_card.pack(fill="x", pady=(0, 8))

ctk.CTkLabel(
    brand_card,
    text="🚗 AI DRIVE",
    font=("Segoe UI", 26, "bold"),
    text_color="#38bdf8",
).pack(anchor="w", padx=18, pady=(16, 6))

ctk.CTkLabel(
    brand_card,
    text="Driver Monitoring Dashboard",
    font=("Segoe UI", 12),
    text_color="#94a3b8",
).pack(anchor="w", padx=18, pady=(0, 16))


# Stats row
stats_row = ctk.CTkFrame(sidebar, fg_color="transparent")
stats_row.pack(fill="x", pady=(0, 8))

time_card = ctk.CTkFrame(
    stats_row,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
time_card.pack(side="left", expand=True, fill="both", padx=(0, 6))

ctk.CTkLabel(
    time_card,
    text="TIME",
    font=("Segoe UI", 11, "bold"),
    text_color="#94a3b8",
).pack(anchor="w", padx=14, pady=(12, 2))

timer_label = ctk.CTkLabel(
    time_card,
    text="00:00",
    font=("Segoe UI", 24, "bold"),
    text_color="#38bdf8",
)
timer_label.pack(anchor="w", padx=14, pady=(0, 12))


alert_card = ctk.CTkFrame(
    stats_row,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
alert_card.pack(side="left", expand=True, fill="both", padx=(6, 0))

ctk.CTkLabel(
    alert_card,
    text="ALERTS",
    font=("Segoe UI", 11, "bold"),
    text_color="#94a3b8",
).pack(anchor="w", padx=14, pady=(12, 2))

count_label = ctk.CTkLabel(
    alert_card,
    text="0",
    font=("Segoe UI", 24, "bold"),
    text_color="#facc15",
)
count_label.pack(anchor="w", padx=14, pady=(0, 12))


# Model card
model_card = ctk.CTkFrame(
    sidebar,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
model_card.pack(fill="x", pady=(0, 8))

ctk.CTkLabel(
    model_card,
    text="MODEL",
    font=("Segoe UI", 11, "bold"),
    text_color="#94a3b8",
).pack(anchor="w", padx=14, pady=(12, 4))

model_label = ctk.CTkLabel(
    model_card,
    text="EfficientNetB0_FastKAN",
    font=("Segoe UI", 16, "bold"),
    text_color="#22c55e",
    justify="left",
    anchor="w",
)
model_label.pack(anchor="w", padx=14, pady=(0, 12))

# AI Analysis card
ai_card = ctk.CTkFrame(
    sidebar,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
ai_card.pack(fill="x", pady=(0, 8))

ctk.CTkLabel(
    ai_card,
    text="AI ANALYSIS",
    font=("Segoe UI", 11, "bold"),
    text_color="#94a3b8",
).pack(anchor="w", padx=14, pady=(12, 6))

ai_info_label = ctk.CTkLabel(
    ai_card,
    text="Label: waiting...\nConf: --\nState: --",
    font=("Consolas", 15, "bold"),
    text_color="#38bdf8",
    justify="left",
    anchor="w",
)
ai_info_label.pack(anchor="w", padx=14, pady=(0, 12))

# Rule Metrics card
rule_card = ctk.CTkFrame(
    sidebar,
    fg_color="#101a33",
    corner_radius=CARD_RADIUS,
    border_width=1,
    border_color="#1e293b",
)
rule_card.pack(fill="x", pady=(0, 8))

ctk.CTkLabel(
    rule_card,
    text="RULE METRICS",
    font=("Segoe UI", 11, "bold"),
    text_color="#94a3b8",
).pack(anchor="w", padx=14, pady=(12, 6))

rule_info_label = ctk.CTkLabel(
    rule_card,
    text="EAR: --\nMAR: --\nHEAD: --",
    font=("Consolas", 15, "bold"),
    text_color="#facc15",
    justify="left",
    anchor="w",
)
rule_info_label.pack(anchor="w", padx=14, pady=(0, 12))

# Buttons
button_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
button_frame.pack(fill="x", side="bottom", pady=(4, 0))

ctk.CTkButton(
    button_frame,
    text="BẮT ĐẦU",
    height=40,
    width=220,
    fg_color="#10b981",
    hover_color="#059669",
    corner_radius=14,
    font=("Segoe UI", 16, "bold"),
    command=start_camera,
).pack(fill="x", pady=(0, 6))

ctk.CTkButton(
    button_frame,
    text="DỪNG",
    height=40,
    width=220,
    fg_color="#3b82f6",
    hover_color="#2563eb",
    corner_radius=14,
    font=("Segoe UI", 16, "bold"),
    command=stop_camera,
).pack(fill="x", pady=6)

ctk.CTkButton(
    button_frame,
    text="THOÁT",
    height=40,
    width=220,
    fg_color="#ef4444",
    hover_color="#dc2626",
    corner_radius=14,
    font=("Segoe UI", 16, "bold"),
    command=exit_app,
).pack(fill="x", pady=(6, 0))


# Nút giao diện
ctk.CTkButton(
    left,
    text="BẮT ĐẦU",
    height=45,
    width=180,
    fg_color="#10b981",
    hover_color="#059669",
    font=("Segoe UI", 15, "bold"),
    command=start_camera,
).pack(pady=(25, 10))

ctk.CTkButton(
    left,
    text="DỪNG",
    height=45,
    width=180,
    fg_color="#3b82f6",
    hover_color="#2563eb",
    font=("Segoe UI", 15, "bold"),
    command=stop_camera,
).pack(pady=10)

ctk.CTkButton(
    left,
    text="THOÁT",
    height=45,
    width=180,
    fg_color="#ef4444",
    hover_color="#dc2626",
    font=("Segoe UI", 15, "bold"),
    command=exit_app,
).pack(pady=10)


# Khu vực bên phải
top_frame = ctk.CTkFrame(right, fg_color="transparent")
top_frame.pack(fill="x", padx=30, pady=(30, 10))

ctk.CTkLabel(
    top_frame,
    text="Driver Drowsiness Detection System",
    font=("Segoe UI", 30, "bold"),
    text_color="white",
).pack(side="left")

status_big = ctk.CTkLabel(
    top_frame,
    text="STOPPED",
    font=("Segoe UI", 28, "bold"),
    text_color="#ff4d4d",
)
status_big.pack(side="right")

camera_frame = ctk.CTkFrame(right, corner_radius=25, fg_color="#1e293b")
camera_frame.pack(padx=30, pady=20, fill="both", expand=True)

camera_label = ctk.CTkLabel(camera_frame, text="")
camera_label.pack(expand=True, padx=20, pady=20)

info_label = ctk.CTkLabel(
    right,
    text="Priority: AI warning → EAR → MAR → Head Down",
    font=("Segoe UI", 14),
    text_color="#94a3b8",
)
info_label.pack(pady=(0, 20))


# MediaPipe + Camera
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

cap = cv2.VideoCapture(CAMERA_INDEX)

if ENABLE_ALARM_SOUND and os.path.exists(ALARM_SOUND_PATH):
    pygame.mixer.music.load(ALARM_SOUND_PATH)
elif ENABLE_ALARM_SOUND:
    print(f"Khong tim thay file am thanh: {ALARM_SOUND_PATH}")


# Detector gục đầu
head_detector = HeadDownDetector(
    calibration_frames=HEAD_CALIBRATION_FRAMES,
    score_delta=HEAD_DOWN_SCORE_DELTA,
    down_time=HEAD_DOWN_TIME,
)

# Load CNN model
cnn_predictor = None

if ENABLE_CNN_MODEL and CNNPredictor is not None:
    try:
        cnn_predictor = CNNPredictor()
        model_label.configure(text=f"Model: {MODEL_TYPE}", text_color="#22c55e")
        print(f"Da load CNN model: {CNN_MODEL_PATH}")
    except Exception as exc:
        cnn_predictor = None
        model_label.configure(text="Model: OFF", text_color="#f97316")
        print(f"Khong load duoc CNN model, dashboard se chay rule-based: {exc}")
else:
    model_label.configure(text="Model: OFF", text_color="#f97316")


# Hàm phụ
def get_cnn_text(result):
    if cnn_predictor is None:
        return "CNN: OFF"

    if result is None:
        return "CNN: waiting..."

    label = result["label"]
    conf = result["confidence"]
    count = result["drowsy_count"]
    size = result["history_size"]
    alert = result["alert"]

    if label == "yawning":
        return f"CNN: yawning {conf:.2f} | MAR check"

    alert_text = "ALERT" if alert else "NO ALERT"
    return f"CNN: {label} {conf:.2f} ({count}/{size}) {alert_text}"


def get_risk_score(status):
    if status == "NORMAL":
        return 20
    if status == "EYES CLOSED":
        return 50
    if status == "YAWNING":
        return 70
    if status == "HEAD DOWN":
        return 80
    if status == "DROWSY":
        return 100
    if status == "AI DROWSY":
        return 100
    if status == "NO FACE":
        return 0
    return 0


def set_status(status):
    if status in ["DROWSY", "AI DROWSY"]:
        color = "#ff4d4d"
    elif status == "YAWNING":
        color = "#facc15"
    elif status == "HEAD DOWN":
        color = "#fb7185"
    elif status == "EYES CLOSED":
        color = "#f97316"
    elif status == "NO FACE":
        color = "#94a3b8"
    else:
        color = "#00ff99"

    status_big.configure(text=status, text_color=color)


def update_timer_and_counts(status):
    global drowsy_count, yawn_count, head_count, ai_count, last_status

    if status != last_status:
        if status == "DROWSY":
            drowsy_count += 1
        elif status == "YAWNING":
            yawn_count += 1
        elif status == "HEAD DOWN":
            head_count += 1
        elif status == "AI DROWSY":
            ai_count += 1

        last_status = status

    total_alerts = drowsy_count + yawn_count + head_count + ai_count
    count_label.configure(text=f"{total_alerts}")

    if status == "AI DROWSY":
        pygame.mixer.music.set_volume(1.0)
        status_big.configure(text="AI WARNING!", text_color="#ff4d4d")
        play_alarm()

    elif status == "DROWSY":
        pygame.mixer.music.set_volume(1.0)
        status_big.configure(text="DANGER!!!", text_color="#ff4d4d")
        play_alarm()

    elif status == "HEAD DOWN":
        pygame.mixer.music.set_volume(0.7)
        status_big.configure(text="HEAD DOWN!", text_color="#fb7185")
        play_alarm()

    elif status == "YAWNING":
        pygame.mixer.music.set_volume(0.4)
        status_big.configure(text="YAWNING!", text_color="#facc15")
        play_alarm()

    else:
        stop_alarm()

    if trip_start_time is not None and is_running:
        elapsed = int(time.time() - trip_start_time)
        timer_label.configure(text=f"{elapsed // 60:02d}:{elapsed % 60:02d}")

def draw_smart_alert(frame, status):
    global blink_flag

    blink_flag = not blink_flag

    alert_text = ""
    color = (0, 255, 0)

    if status == "AI DROWSY":
        alert_text = "AI WARNING!"
        color = (0, 0, 255)

    elif status == "DROWSY":
        alert_text = "WAKE UP!"
        color = (0, 0, 255)

    elif status == "YAWNING":
        alert_text = "YAWN DETECTED!"
        color = (0, 0, 255)

    elif status == "HEAD DOWN":
        alert_text = "FOCUS ON DRIVING!"
        color = (0, 100, 255)

    if alert_text and blink_flag:
        cv2.putText(
            frame,
            alert_text,
            (260, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            color,
            4,
        )

def reset_head_calibration_event(event=None):
    global cnn_result, frame_count

    head_detector.reset()
    frame_count = 0
    cnn_result = None

    if cnn_predictor is not None:
        cnn_predictor.history.clear()

    print("Da reset calibration guc dau va CNN history")


app.bind("r", reset_head_calibration_event)
app.bind("R", reset_head_calibration_event)


def update_ai_info(result):
    if result is None:
        ai_info_label.configure(
            text="Label : waiting...\nConf  : --\nState : --",
            text_color="#94a3b8",
        )
        return

    label = result["label"]
    conf = result["confidence"]
    alert = result["alert"]

    if label == "yawning":
        state_text = "MAR CHECK"
    else:
        state_text = "ALERT" if alert else "NO ALERT"

    if label in ["sleepyCombination", "slowBlinkWithNodding"]:
        color = "#ef4444"
    elif label == "yawning":
        color = "#facc15"
    else:
        color = "#38bdf8"

    ai_info_label.configure(
        text=(
            f"Label : {label}\n"
            f"Conf  : {conf:.2f}\n"
            f"State : {state_text}"
        ),
        text_color=color,
    )

def update_rule_info(ear, mar, head_delta, status):
    if status in ["DROWSY", "EYES CLOSED"]:
        color = "#f97316"
    elif status == "YAWNING":
        color = "#facc15"
    elif status == "HEAD DOWN":
        color = "#fb7185"
    else:
        color = "#22c55e"

    rule_info_label.configure(
        text=(
            f"EAR  : {ear:.2f}\n"
            f"MAR  : {mar:.2f}\n"
            f"HEAD : {head_delta:.3f}"
        ),
        text_color=color,
    )

# Update camera frame
def update_frame():
    global closed_start_time, frame_count, cnn_result

    ret, frame = cap.read()

    if not ret:
        app.after(30, update_frame)
        return

    frame_count += 1
    frame = cv2.flip(frame, 1)

    if not is_running:
        frame = cv2.resize(
            frame,
            (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT),
            interpolation=cv2.INTER_CUBIC,
        )
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        app.after(30, update_frame)
        return

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

        # 1. AI MODEL DỰ ĐOÁN TRƯỚC
        ai_label = None
        ai_conf = 0.0

        if cnn_predictor is not None:
            try:
                if frame_count % CNN_PREDICT_EVERY_N_FRAMES == 0:
                    cnn_result = cnn_predictor.predict(frame, face)

                if cnn_result is not None:
                    ai_label = cnn_result["label"]
                    ai_conf = cnn_result["confidence"]

            except Exception as exc:
                print(f"Loi khi CNN predict: {exc}")
                cnn_result = None
                ai_label = None
                ai_conf = 0.0

        # 2. TÍNH RULE-BASED ĐỂ DỰ PHÒNG
        eye_alert = False

        if ear < EAR_THRESHOLD:
            if closed_start_time is None:
                closed_start_time = time.time()

            if time.time() - closed_start_time >= CLOSED_EYES_TIME:
                eye_alert = True
        else:
            closed_start_time = None

        yawn_alert = mar > MAR_THRESHOLD

        is_head_alert, head_score, head_delta, is_calibrated = head_detector.update(face)

        # 3. QUYẾT ĐỊNH TRẠNG THÁI
        if (
            ai_label in ["sleepyCombination", "slowBlinkWithNodding"]
            and ai_conf >= CNN_CONFIDENCE_THRESHOLD
        ):
            status = "AI DROWSY"

        elif (
            ai_label == "yawning"
            and ai_conf >= AI_YAWNING_CONF_THRESHOLD
            and mar >= AI_YAWNING_MAR_THRESHOLD
        ):
            status = "YAWNING"

        elif eye_alert:
            status = "DROWSY"

        elif is_head_alert:
            status = "HEAD DOWN"

        elif yawn_alert:
            status = "YAWNING"

        elif ear < EAR_THRESHOLD:
            status = "EYES CLOSED"

        else:
            status = "NORMAL"

    else:
        status = "NO FACE"
        closed_start_time = None
        head_detector.reset_timer()

    # Hiển thị debug CNN trên camera
    if SHOW_CAMERA_DEBUG_TEXT:
        cv2.putText(
            frame,
            get_cnn_text(cnn_result),
            (CNN_TEXT_X, CNN_TEXT_Y),
            cv2.FONT_HERSHEY_SIMPLEX,
            CNN_TEXT_SCALE,
            (0, 255, 255),
            CNN_TEXT_THICKNESS,
        )

        if cnn_result is not None:
            y0 = CNN_PROB_START_Y

            for idx, (name, prob) in enumerate(zip(CNN_CLASS_NAMES, cnn_result["probs"])):
                cv2.putText(
                    frame,
                    f"{name}: {prob:.2f}",
                    (CNN_TEXT_X, y0 + idx * CNN_PROB_LINE_GAP),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    CNN_PROB_TEXT_SCALE,
                    (255, 255, 0),
                    CNN_PROB_TEXT_THICKNESS,
                )

            cv2.putText(
                frame,
                f"MAR: {mar:.2f} | AI_YAWN_MAR: {AI_YAWNING_MAR_THRESHOLD:.2f}",
                (CNN_TEXT_X, y0 + len(CNN_CLASS_NAMES) * CNN_PROB_LINE_GAP + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                CNN_PROB_TEXT_SCALE,
                (0, 255, 255),
                CNN_PROB_TEXT_THICKNESS,
            )

            if cnn_result["label"] == "yawning" and mar < AI_YAWNING_MAR_THRESHOLD:
                cv2.putText(
                    frame,
                    "AI yawning blocked: mouth not open enough",
                    (CNN_TEXT_X, y0 + len(CNN_CLASS_NAMES) * CNN_PROB_LINE_GAP + 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    CNN_PROB_TEXT_SCALE,
                    (0, 0, 255),
                    CNN_PROB_TEXT_THICKNESS,
                )


    update_ai_info(cnn_result)
    update_rule_info(ear, mar, head_delta, status)

    set_status(status)
    update_timer_and_counts(status)
    draw_smart_alert(frame, status)
    

    frame = cv2.resize(
        frame,
        (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    app.after(10, update_frame)


update_frame()
app.protocol("WM_DELETE_WINDOW", exit_app)
app.mainloop()