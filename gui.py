import customtkinter as ctk
import subprocess
import sys

process = None

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

def start_system():
    global process
    if process is None:
        process = subprocess.Popen([sys.executable, "main.py"])

def stop_system():
    global process
    if process is not None:
        process.terminate()
        process = None

def exit_app():
    stop_system()
    app.destroy()

# ===== REALTIME STATUS =====
def update_status():
    try:
        with open("status.txt", "r") as f:
            s = f.read()

            if "DROWSINESS" in s:
                status_value.configure(text="DROWSY", text_color="#ff4d4d")
                progress.set(1)
            elif "YAWNING" in s:
                status_value.configure(text="YAWNING", text_color="#facc15")
                progress.set(0.7)
            elif "HEAD" in s:
                status_value.configure(text="HEAD DOWN", text_color="#fb7185")
                progress.set(0.8)
            else:
                status_value.configure(text="NORMAL", text_color="#00ff99")
                progress.set(0.2)

    except:
        pass

    app.after(500, update_status)

# ===== UI =====
app = ctk.CTk()
app.title("AI Driver Dashboard")
app.geometry("900x550")

sidebar = ctk.CTkFrame(app, width=220)
sidebar.pack(side="left", fill="y")

main = ctk.CTkFrame(app)
main.pack(side="right", fill="both", expand=True)

ctk.CTkLabel(sidebar, text="🚗 AI DRIVE",
             font=("Segoe UI", 24, "bold")).pack(pady=30)

ctk.CTkButton(sidebar, text="Start",
              command=start_system).pack(pady=10)

ctk.CTkButton(sidebar, text="Stop",
              command=stop_system).pack(pady=10)

ctk.CTkButton(sidebar, text="Exit",
              command=exit_app).pack(pady=10)

ctk.CTkLabel(main,
             text="Driver Drowsiness System",
             font=("Segoe UI", 28, "bold")).pack(pady=20)

status_value = ctk.CTkLabel(main,
                            text="OFFLINE",
                            font=("Segoe UI", 36, "bold"),
                            text_color="#ff4d4d")
status_value.pack(pady=20)

progress = ctk.CTkProgressBar(main, width=400)
progress.pack(pady=10)
progress.set(0)

update_status()

app.protocol("WM_DELETE_WINDOW", exit_app)
app.mainloop()