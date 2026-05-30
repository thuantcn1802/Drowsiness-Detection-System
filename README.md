# Driver Drowsiness Detection System

## 1. Overview

**Driver Drowsiness Detection System** is a real-time AI-based driver monitoring system designed to detect early signs of driver fatigue and drowsiness through a webcam.

The system combines two approaches:

* **Deep Learning-based classification** using an `EfficientNetB0_FastKAN` model.
* **Rule-based facial metric analysis** using MediaPipe Face Mesh landmarks.

The system can detect several fatigue-related behaviors, including:

* Prolonged eye closure
* Yawning
* Head-down posture
* General drowsiness patterns predicted by the AI model

When an abnormal state is detected, the dashboard displays a visual warning and plays an alarm sound.

---

## 2. Main Features

### 2.1 Real-time Camera Monitoring

The system captures video frames from a webcam using OpenCV and processes them in real time.

### 2.2 AI-based Drowsiness Classification

The CNN model predicts one of the following driver states:

| Class                  | Description                              |
| ---------------------- | ---------------------------------------- |
| `notdrowsy`            | Normal / alert state                     |
| `sleepyCombination`    | Combined sleepy facial expression        |
| `slowBlinkWithNodding` | Slow blinking combined with head nodding |
| `yawning`              | Yawning behavior                         |

The AI model is used to detect high-level facial patterns related to drowsiness.

### 2.3 Eye Closure Detection using EAR

The system calculates the **Eye Aspect Ratio (EAR)** from eye landmarks.

If the EAR value remains below the configured threshold for a specific duration, the driver is considered drowsy.

Default configuration:

```python
EAR_THRESHOLD = 0.22
CLOSED_EYES_TIME = 1.5
```

### 2.4 Yawning Detection using MAR

The system calculates the **Mouth Aspect Ratio (MAR)** from mouth landmarks.

Yawning can be detected by:

1. Rule-based MAR threshold.
2. AI prediction `yawning` confirmed by MAR.

This design helps reduce false positives when the AI model predicts yawning while the mouth is not actually open.

Default configuration:

```python
MAR_THRESHOLD = 0.45
AI_YAWNING_CONF_THRESHOLD = 0.85
AI_YAWNING_MAR_THRESHOLD = 0.50
```

When yawning is detected, the system displays:

```text
Status: YAWNING
Camera alert: YAWN DETECTED!
```

### 2.5 Head-down Detection

The system detects head-down posture by comparing the nose position with the eye region.

A short calibration phase is performed when monitoring starts. The system stores the normal head position and then detects whether the head moves downward beyond a threshold.

Default configuration:

```python
HEAD_CALIBRATION_FRAMES = 30
HEAD_DOWN_SCORE_DELTA = 0.025
HEAD_DOWN_TIME = 1.5
```

### 2.6 Audio Warning

The system plays an alarm sound when a warning state is detected.

Default alarm path:

```python
ALARM_SOUND_PATH = "sounds/alarm.mp3"
```

---

## 3. Technology Stack

| Technology             | Purpose                             |
| ---------------------- | ----------------------------------- |
| Python                 | Main programming language           |
| OpenCV                 | Camera capture and frame processing |
| MediaPipe Face Mesh    | Facial landmark detection           |
| PyTorch                | Deep learning model inference       |
| EfficientNetB0_FastKAN | Drowsiness classification model     |
| CustomTkinter          | Graphical dashboard interface       |
| Pygame                 | Alarm sound playback                |
| PIL / Pillow           | Image conversion for GUI display    |
| NumPy                  | Numerical calculation               |

---

## 4. Project Structure

```text
## Project Structure

driver-drowsiness-system/
├── src/
│   ├── app/              # GUI and application entry points
│   ├── config/           # System configuration
│   ├── detection/        # EAR, MAR, head pose, CNN predictor
│   ├── models/           # Deep learning model architecture
│   └── utils/            # Utility functions
├── assets/sounds/        # Alarm sound files
├── weights/              # Trained model weights
├── debug/                # Debug outputs
├── tests/                # Test scripts
├── README.md
└── requirements.txt
```

### File Description

| File / Folder                     | Description                                                               |
| --------------------------------- | ------------------------------------------------------------------------- |
| `config.py`                       | Stores system configuration, thresholds, GUI settings, and model settings |
| `gui_camera.py`                   | Main dashboard application with real-time camera monitoring               |
| `cnn_predictor.py`                | Loads the trained AI model and performs prediction                        |
| `model.py`                        | Defines the `EfficientNetB0_FastKAN` model architecture                   |
| `eye.py`                          | Calculates EAR for eye closure detection                                  |
| `yawn.py`                         | Calculates MAR for yawning detection                                      |
| `head_pose.py`                    | Detects head-down posture using facial landmarks                          |
| `requirements.txt`                | Lists required Python packages                                            |
| `best_EfficientNetB0_FastKAN.pth` | Trained model weight file                                                 |
| `sounds/alarm.mp3`                | Alarm sound file                                                          |
| `debug_cnn/`                      | Stores debug images of cropped face inputs for CNN prediction             |

---

## 5. System Workflow

The system works according to the following pipeline:

```text
Webcam
→ OpenCV reads frame
→ MediaPipe detects facial landmarks
→ EAR, MAR, and Head score are calculated
→ Face region is cropped for CNN prediction
→ AI model predicts driver state
→ Rule-based and AI results are combined
→ Dashboard is updated
→ Visual and audio alerts are triggered
```

---

## 6. Warning Decision Logic

The final warning state is decided by combining AI model output and rule-based metrics.

Current priority logic:

```text
1. AI detects sleepyCombination or slowBlinkWithNodding
   → AI DROWSY

2. AI detects yawning + MAR confirms mouth opening
   → YAWNING

3. EAR detects prolonged eye closure
   → DROWSY

4. Head score detects head-down posture
   → HEAD DOWN

5. MAR detects yawning
   → YAWNING

6. EAR is low but not long enough
   → EYES CLOSED

7. No abnormal sign
   → NORMAL
```

### Why combine AI and rule-based methods?

The AI model can recognize high-level facial patterns, but it may be affected by lighting, camera angle, or training data limitations.

Rule-based metrics are used to strengthen detection for specific signs:

| Metric     | Purpose                        |
| ---------- | ------------------------------ |
| EAR        | Detect eye closure             |
| MAR        | Detect mouth opening / yawning |
| Head score | Detect head-down posture       |

This hybrid approach improves system reliability and reduces false warnings.

---

## 7. Dashboard Interface

The dashboard contains two main sections.

### 7.1 Left Sidebar

The sidebar displays:

* Driving time
* Total number of alerts
* AI model information
* AI analysis result
* Rule-based metrics

Example:

```text
AI Analysis
Label : yawning
Conf  : 0.88
State : MAR CHECK

Rule Metrics
EAR  : 0.31
MAR  : 0.62
HEAD : 0.004
```

### 7.2 Camera Panel

The camera panel displays:

* Real-time driver video
* Main warning label
* On-camera warning message

Warning messages:

| Status      | Camera Message      |
| ----------- | ------------------- |
| `AI DROWSY` | `AI WARNING!`       |
| `DROWSY`    | `WAKE UP!`          |
| `YAWNING`   | `YAWN DETECTED!`    |
| `HEAD DOWN` | `FOCUS ON DRIVING!` |

---

## 8. Installation

### 8.1 Create Python Environment

Python 3.10 is recommended.

Using Conda:

```bash
conda create -n DACN1 python=3.10
conda activate DACN1
```

Using virtualenv:

```bash
python -m venv venv
venv\Scripts\activate
```

### 8.2 Install Dependencies

```bash
pip install -r requirements.txt
```

If PyTorch installation fails, install it separately:

```bash
pip install torch torchvision
```

---

## 9. Required Files

Before running the system, make sure the following files exist:

```text
best_EfficientNetB0_FastKAN.pth
sounds/alarm.mp3
```

If the model file is missing, the AI model cannot be loaded.

If the alarm file is missing, the system can still run but audio warning will not work.

---

## 10. How to Run

Run the dashboard:

```bash
python -m src.app.gui_camera
```

Then:

1. Click **BẮT ĐẦU** to start monitoring.
2. Sit normally for a few seconds so the system can calibrate the head position.
3. Perform testing actions such as yawning, closing eyes, or lowering the head.
4. Click **DỪNG** to pause monitoring.
5. Click **THOÁT** to close the application.

---

## 11. Important Configuration

Most system parameters are defined in `config.py`.

### 11.1 Eye Closure

```python
EAR_THRESHOLD = 0.22
CLOSED_EYES_TIME = 1.5
```

| Parameter          | Meaning                          |
| ------------------ | -------------------------------- |
| `EAR_THRESHOLD`    | Eye closure threshold            |
| `CLOSED_EYES_TIME` | Required duration before warning |

### 11.2 Yawning

```python
MAR_THRESHOLD = 0.45
AI_YAWNING_CONF_THRESHOLD = 0.85
AI_YAWNING_MAR_THRESHOLD = 0.50
```

| Parameter                   | Meaning                             |
| --------------------------- | ----------------------------------- |
| `MAR_THRESHOLD`             | Rule-based yawning threshold        |
| `AI_YAWNING_CONF_THRESHOLD` | Minimum AI confidence for yawning   |
| `AI_YAWNING_MAR_THRESHOLD`  | MAR threshold to confirm AI yawning |

### 11.3 Head-down Detection

```python
HEAD_CALIBRATION_FRAMES = 30
HEAD_DOWN_SCORE_DELTA = 0.025
HEAD_DOWN_TIME = 1.5
```

| Parameter                 | Meaning                               |
| ------------------------- | ------------------------------------- |
| `HEAD_CALIBRATION_FRAMES` | Number of frames used for calibration |
| `HEAD_DOWN_SCORE_DELTA`   | Head-down sensitivity                 |
| `HEAD_DOWN_TIME`          | Required duration before warning      |

### 11.4 CNN Model

```python
CNN_CONFIDENCE_THRESHOLD = 0.70
CNN_HISTORY_SIZE = 7
CNN_DROWSY_THRESHOLD = 5
CNN_PREDICT_EVERY_N_FRAMES = 3
```

| Parameter                    | Meaning                                  |
| ---------------------------- | ---------------------------------------- |
| `CNN_CONFIDENCE_THRESHOLD`   | Minimum confidence for AI drowsy classes |
| `CNN_HISTORY_SIZE`           | Prediction history length                |
| `CNN_DROWSY_THRESHOLD`       | Number of drowsy predictions required    |
| `CNN_PREDICT_EVERY_N_FRAMES` | CNN prediction frequency                 |

---

## 12. Debug Mode

The system supports CNN debug mode.

Configuration:

```python
ENABLE_CNN_DEBUG = True
CNN_DEBUG_SAVE_IMAGE = True
CNN_DEBUG_PRINT_PROBS = True
CNN_DEBUG_DIR = "debug_cnn"
CNN_DEBUG_SAVE_EVERY_N = 10
```

When enabled:

* The system prints class probabilities in the terminal.
* Cropped face images used by the CNN model are saved to `debug_cnn/`.

Example terminal output:

```text
========== CNN DEBUG ==========
Predict #20
Top label: yawning
Confidence: 0.8821
History: 0/7
Alert: False
Probabilities:
  notdrowsy: 0.0712
  sleepyCombination: 0.0221
  slowBlinkWithNodding: 0.0246
  yawning: 0.8821
================================
```

Debug mode is useful for checking:

* Whether the face crop is correct.
* Whether the model is confused between classes.
* Whether lighting or camera angle affects prediction.
* Whether yawning is falsely detected.

For final demonstration, debug image saving can be disabled:

```python
ENABLE_CNN_DEBUG = False
CNN_DEBUG_SAVE_IMAGE = False
CNN_DEBUG_PRINT_PROBS = False
```

---

## 13. Demo Scenario

This section describes a recommended demonstration flow for presentation.

### Step 1: Start the System

Run:

```bash
python gui_camera.py
```

Click **BẮT ĐẦU**.

Presentation script:

```text
This is the real-time driver drowsiness detection dashboard.
The system uses a webcam to monitor the driver and combines AI prediction with facial metrics such as EAR, MAR, and head pose.
```

### Step 2: Normal State

Action:

* Sit upright.
* Look directly at the camera.
* Keep eyes open and mouth closed.

Expected result:

```text
Status: NORMAL
AI label: notdrowsy
EAR: normal
MAR: low
Head: stable
```

Presentation script:

```text
In the normal state, the system recognizes that the driver is alert.
The AI model predicts notdrowsy and all rule-based metrics remain within safe thresholds.
```

### Step 3: Yawning Detection

Action:

* Open mouth clearly as if yawning.
* Hold for 2–3 seconds.

Expected result:

```text
Status: YAWNING
Camera message: YAWN DETECTED!
MAR increases
Alarm sound plays at a low warning level
```

Presentation script:

```text
When the driver yawns, the MAR value increases.
If the AI model predicts yawning, the system also verifies it using MAR to reduce false alarms.
After confirmation, the system displays YAWNING and shows the message YAWN DETECTED on the camera.
```

### Step 4: Eye Closure Detection

Action:

* Close eyes continuously for around 2 seconds.

Expected result:

```text
Status: EYES CLOSED
Then: DROWSY
Camera message: WAKE UP!
Alarm sound plays strongly
```

Presentation script:

```text
When the driver's eyes remain closed for longer than the configured threshold, the EAR value stays low.
The system then changes the state to DROWSY and triggers a stronger warning.
```

### Step 5: Head-down Detection

Action:

* Sit normally for calibration.
* Lower the head and keep it down for around 2 seconds.

Expected result:

```text
Status: HEAD DOWN
Camera message: FOCUS ON DRIVING!
Head metric increases
Alarm sound plays
```

Presentation script:

```text
The system first calibrates the normal head position.
When the head moves downward and remains in that position, the head score changes and the system warns the driver to focus on driving.
```

### Step 6: AI Drowsy Detection

Action:

* Slowly blink.
* Slightly nod the head.
* Show a tired facial expression.

Expected result:

```text
Status: AI DROWSY
Camera message: AI WARNING!
AI label: sleepyCombination or slowBlinkWithNodding
```

Presentation script:

```text
For complex drowsiness patterns, the AI model is used to recognize facial states such as sleepyCombination or slowBlinkWithNodding.
When confidence is high enough, the system triggers AI DROWSY.
```

### Step 7: Conclusion

Presentation script:

```text
The demo shows that the system can detect different signs of driver fatigue in real time.
By combining AI prediction with EAR, MAR, and head-pose metrics, the system can provide more stable and reliable warnings.
This approach can be further developed for real vehicle environments or embedded devices.
```

---

## 14. Troubleshooting

### 14.1 Camera Not Opening

Check the camera index in `config.py`:

```python
CAMERA_INDEX = 0
```

If the system cannot open the camera, try:

```python
CAMERA_INDEX = 1
```

### 14.2 Model File Not Found

Make sure this file exists in the project root:

```text
best_EfficientNetB0_FastKAN.pth
```

### 14.3 Alarm Sound Not Found

Make sure this file exists:

```text
sounds/alarm.mp3
```

### 14.4 CustomTkinter Image Warning

You may see this warning:

```text
CTkLabel Warning: Given image is not CTkImage
```

This is only a display warning. It does not stop the system.

### 14.5 AI Incorrectly Detects Yawning

Possible causes:

* The mouth is slightly open.
* Lighting is poor.
* Face crop is not stable.
* The model is sensitive to the yawning class.

Recommended fixes:

```python
AI_YAWNING_CONF_THRESHOLD = 0.90
AI_YAWNING_MAR_THRESHOLD = 0.55
```

### 14.6 Yawning Is Not Detected

If real yawning is not detected, reduce the thresholds slightly:

```python
AI_YAWNING_CONF_THRESHOLD = 0.80
AI_YAWNING_MAR_THRESHOLD = 0.45
```

---

## 15. Limitations

The current system still has several limitations:

* Performance depends on webcam quality.
* Low light or backlight may reduce accuracy.
* The AI model may not detect closed eyes well if the dataset does not contain a separate `closedEyes` class.
* The system is currently designed for webcam-based desktop demonstration.
* It has not yet been deployed on an embedded device or real vehicle hardware.
* Long-term alert history is not stored in a database.

---

## 16. Future Development

Possible improvements:

* Add a dedicated `closedEyes` class to the AI model.
* Fine-tune the model using real webcam data.
* Improve face crop stability.
* Store alert history in a database.
* Add trip reports and driver fatigue statistics.
* Deploy the system on embedded devices such as Jetson Nano or Raspberry Pi.
* Integrate mobile notification or vehicle alert systems.
* Optimize inference speed for real-time deployment.

---

## 17. Conclusion

The Driver Drowsiness Detection System provides a real-time solution for detecting early signs of driver fatigue. It combines AI-based facial state classification with rule-based facial metrics to detect yawning, prolonged eye closure, and head-down posture.

This hybrid approach improves reliability and makes the system suitable for demonstration, research, and further development toward real-world driver safety applications.
