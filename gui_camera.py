import customtkinter as ctk
import cv2
import mediapipe as mp
from PIL import Image, ImageTk
import time
import pygame

from config import *
from eye import calculate_ear
from yawn import calculate_mar
from head_pose import is_head_down

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

pygame.mixer.init()
alarm_playing = False

app = ctk.CTk()
app.geometry("1300x760")
app.title("AI Driver Dashboard")

is_running = False
closed_start_time = None
trip_start_time = None
blink_flag = False

drowsy_count = 0
yawn_count = 0
head_count = 0
last_status = "NORMAL"

left = ctk.CTkFrame(app, width=260, fg_color="#111827")
left.pack(side="left", fill="y")

right = ctk.CTkFrame(app, fg_color="#0f172a")
right.pack(side="right", expand=True, fill="both")

ctk.CTkLabel(left, text="🚗 AI DRIVE",
             font=("Segoe UI", 28, "bold"),
             text_color="#38bdf8").pack(pady=(35, 15))

status_label = ctk.CTkLabel(left, text="STOPPED",
                            font=("Segoe UI", 24, "bold"),
                            text_color="#ff4d4d")
status_label.pack(pady=12)

progress = ctk.CTkProgressBar(left, width=200)
progress.pack(pady=8)
progress.set(0)

risk_label = ctk.CTkLabel(left, text="Risk: 0%",
                          font=("Segoe UI", 19, "bold"),
                          text_color="#38bdf8")
risk_label.pack(pady=10)

timer_label = ctk.CTkLabel(left, text="Time: 00:00",
                           font=("Segoe UI", 18, "bold"),
                           text_color="#38bdf8")
timer_label.pack(pady=8)

count_label = ctk.CTkLabel(left, text="Alerts: 0",
                           font=("Segoe UI", 18, "bold"),
                           text_color="#facc15")
count_label.pack(pady=8)

ctk.CTkLabel(left,
             text="AI Monitoring\nEyes • Yawning • Head Pose",
             font=("Segoe UI", 14),
             text_color="#94a3b8",
             justify="center").pack(pady=15)

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

def start_camera():
    global is_running, trip_start_time
    is_running = True
    if trip_start_time is None:
        trip_start_time = time.time()

def stop_camera():
    global is_running
    is_running = False
    stop_alarm()
    status_label.configure(text="STOPPED", text_color="#ff4d4d")
    status_big.configure(text="STOPPED", text_color="#ff4d4d")
    risk_label.configure(text="Risk: 0%", text_color="#38bdf8")
    progress.set(0)

def exit_app():
    stop_alarm()
    pygame.mixer.quit()
    cap.release()
    face_mesh.close()
    app.destroy()

ctk.CTkButton(left, text="BẮT ĐẦU", height=45, width=180,
              fg_color="#10b981", hover_color="#059669",
              font=("Segoe UI", 15, "bold"),
              command=start_camera).pack(pady=(25, 10))

ctk.CTkButton(left, text="DỪNG", height=45, width=180,
              fg_color="#3b82f6", hover_color="#2563eb",
              font=("Segoe UI", 15, "bold"),
              command=stop_camera).pack(pady=10)

ctk.CTkButton(left, text="THOÁT", height=45, width=180,
              fg_color="#ef4444", hover_color="#dc2626",
              font=("Segoe UI", 15, "bold"),
              command=exit_app).pack(pady=10)

top_frame = ctk.CTkFrame(right, fg_color="transparent")
top_frame.pack(fill="x", padx=30, pady=(30, 10))

ctk.CTkLabel(top_frame,
             text="Driver Drowsiness Detection System",
             font=("Segoe UI", 30, "bold"),
             text_color="white").pack(side="left")

status_big = ctk.CTkLabel(top_frame, text="STOPPED",
                          font=("Segoe UI", 28, "bold"),
                          text_color="#ff4d4d")
status_big.pack(side="right")

camera_frame = ctk.CTkFrame(right, corner_radius=25, fg_color="#1e293b")
camera_frame.pack(padx=30, pady=20, fill="both", expand=True)

camera_label = ctk.CTkLabel(camera_frame, text="")
camera_label.pack(expand=True, padx=20, pady=20)

info_label = ctk.CTkLabel(
    right,
    text="Smart alarm: Yawning nhẹ • Head down vừa • Drowsy/Yawn >= 5 rất to",
    font=("Segoe UI", 14),
    text_color="#94a3b8"
)
info_label.pack(pady=(0, 20))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 78, 308, 82, 312]

NOSE = 1
LEFT_EYE_CENTER = 33
RIGHT_EYE_CENTER = 263

cap = cv2.VideoCapture(0)
pygame.mixer.music.load("sounds/alarm.mp3")

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
    return 0

def set_status(status):
    risk_score = get_risk_score(status)

    if status == "DROWSY":
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

    progress.set(risk_score / 100)
    status_label.configure(text=status, text_color=color)
    status_big.configure(text=status, text_color=color)
    risk_label.configure(text=f"Risk: {risk_score}%", text_color=color)

def update_timer_and_counts(status):
    global drowsy_count, yawn_count, head_count, last_status

    if status != last_status:
        if status == "DROWSY":
            drowsy_count += 1
        elif status == "YAWNING":
            yawn_count += 1
        elif status == "HEAD DOWN":
            head_count += 1
        last_status = status

    total_alerts = drowsy_count + yawn_count + head_count
    count_label.configure(text=f"Alerts: {total_alerts}")

    if yawn_count >= 5:
        pygame.mixer.music.set_volume(1.0)
        status_big.configure(text="PLEASE REST!", text_color="#ff4d4d")
        status_label.configure(text="YAWN WARNING", text_color="#ff4d4d")
        play_alarm()
    elif status == "DROWSY":
        pygame.mixer.music.set_volume(1.0)
        status_big.configure(text="DANGER!!!", text_color="#ff4d4d")
        status_label.configure(text="DROWSY ALERT", text_color="#ff4d4d")
        play_alarm()
    elif status == "HEAD DOWN":
        pygame.mixer.music.set_volume(0.6)
        play_alarm()
    elif status == "YAWNING":
        pygame.mixer.music.set_volume(0.3)
        play_alarm()
    else:
        stop_alarm()

    if trip_start_time is not None and is_running:
        elapsed = int(time.time() - trip_start_time)
        timer_label.configure(text=f"Time: {elapsed//60:02d}:{elapsed%60:02d}")

def draw_smart_alert(frame, status):
    global blink_flag
    blink_flag = not blink_flag

    alert_text = ""
    color = (0, 255, 0)

    if yawn_count >= 5:
        alert_text = "PLEASE REST!"
        color = (0, 0, 255)
    elif status == "YAWNING":
        alert_text = "YOU ARE TIRED!"
        color = (0, 255, 255)
    elif status == "HEAD DOWN":
        alert_text = "FOCUS ON DRIVING!"
        color = (0, 100, 255)
    elif status == "DROWSY":
        alert_text = "WAKE UP!"
        color = (0, 0, 255)

    if alert_text and blink_flag:
        cv2.putText(frame, alert_text, (260, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4)

def update_frame():
    global closed_start_time

    ret, frame = cap.read()
    if not ret:
        app.after(30, update_frame)
        return

    frame = cv2.flip(frame, 1)

    if not is_running:
        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
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

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]

        left_eye = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in RIGHT_EYE]
        mouth = [(int(face.landmark[i].x * w), int(face.landmark[i].y * h)) for i in MOUTH]

        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2
        mar = calculate_mar(mouth)

        if ear < EAR_THRESHOLD:
            if closed_start_time is None:
                closed_start_time = time.time()
            if time.time() - closed_start_time >= CLOSED_EYES_TIME:
                status = "DROWSY"
            else:
                status = "EYES CLOSED"
        else:
            closed_start_time = None

        if mar > MAR_THRESHOLD:
            status = "YAWNING"

        nose_y = face.landmark[NOSE].y
        eye_avg_y = (
            face.landmark[LEFT_EYE_CENTER].y +
            face.landmark[RIGHT_EYE_CENTER].y
        ) / 2

        if is_head_down(nose_y, eye_avg_y, HEAD_DOWN_THRESHOLD):
            status = "HEAD DOWN"

        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    else:
        status = "NO FACE"

    set_status(status)
    update_timer_and_counts(status)
    draw_smart_alert(frame, status)

    frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    app.after(10, update_frame)

update_frame()
app.protocol("WM_DELETE_WINDOW", exit_app)
app.mainloop()