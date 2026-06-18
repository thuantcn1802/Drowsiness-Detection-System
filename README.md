# Driver Drowsiness Detection System

## Overview

Driver Drowsiness Detection System is a real-time AI-powered driver monitoring application that detects signs of fatigue through a webcam.

The system combines:

- Deep Learning (`EfficientNetB0_FastKAN`) for drowsiness classification
- MediaPipe Face Mesh for facial landmark detection
- Rule-based analysis using EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), and head pose estimation
- Real-time warning dashboard with audio alerts

## Features

- Real-time webcam monitoring
- Drowsiness detection using a trained deep learning model
- Eye closure detection (EAR)
- Yawning detection (MAR)
- Head-down posture detection
- Audio and visual alerts
- Desktop GUI built with CustomTkinter
- Debug mode for model prediction analysis

---

## Project Structure

```text
driver-drowsiness-system/
│
├── src/
│   ├── app/
│   ├── config/
│   ├── detection/
│   ├── models/
│   └── utils/
│
├── sounds/
│   └── alarm.mp3
│
├── weights/
│   └── best_EfficientNetB0_FastKAN.pth
│
├── debug/
├── tests/
├── train/
├── README.md
└── requirements.txt
```

---

## Technologies

- Python
- OpenCV
- MediaPipe
- PyTorch
- EfficientNetB0 + FastKAN
- CustomTkinter
- Pillow
- NumPy
- Pygame

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/thuantcn1802/driver-drowsiness-system.git
cd driver-drowsiness-system
```

### 2. Create a virtual environment
Option 1: Conda Environment (Recommended)

Create a new Conda environment:
conda create -n drowsiness python=3.10 -y

Activate the environment:
conda activate drowsiness

Option 2: Python Virtual Environment (venv)

Create a virtual environment:
python -m venv venv

Activate the environment:
Windows
venv\Scripts\activate

Linux / macOS
source venv/bin/activate

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the trained model (Required)

The model file is not included in this repository because of its size.

After cloning the project, download:

**best_EfficientNetB0_FastKAN.pth**

Google Drive:

https://drive.google.com/file/d/1Jo0hj1ZQt9GZ7aDssd63ownw-j7147kq/view?usp=drive_link

Create the `weights` folder if it does not exist:

```bash
mkdir weights
```

Place the downloaded file here:

```text
driver-drowsiness-system/
└── weights/
    └── best_EfficientNetB0_FastKAN.pth
```

Without this file, the AI model cannot be loaded and the application will not start correctly.

---

## Running the Application

```bash
python -m src.app.gui_camera
```

### Usage

1. Start the application.
2. Allow webcam access.
3. Keep your head in a normal position for calibration.
4. Test behaviors such as:
   - Eye closure
   - Yawning
   - Head-down posture
5. Observe real-time warnings and alerts.

---

## Detection Logic

### Eye Closure Detection

Uses Eye Aspect Ratio (EAR):

```python
EAR_THRESHOLD = 0.22
CLOSED_EYES_TIME = 1.5
```

### Yawning Detection

Uses Mouth Aspect Ratio (MAR):

```python
MAR_THRESHOLD = 0.45
```

### Head-Down Detection

Uses calibrated facial landmark positions:

```python
HEAD_DOWN_TIME = 1.5
```

### AI Classification Classes

- notdrowsy
- sleepyCombination
- slowBlinkWithNodding
- yawning

---

## System Workflow

```text
Webcam
   ↓
MediaPipe Face Mesh
   ↓
EAR / MAR / Head Pose Analysis
   ↓
CNN Prediction
   ↓
Decision Fusion
   ↓
Alert Generation
   ↓
GUI Dashboard + Alarm
```

---

## Testing

```bash
python tests/test_camera.py
```

```bash
python tests/test_mediapipe.py
```

---

## Notes

- Python 3.10 is recommended.
- Ensure your webcam is working properly.
- Ensure the model file exists inside the `weights` directory before running the application.
- The alarm sound file must remain inside the `sounds` directory.

## License

This project is intended for educational and research purposes.
