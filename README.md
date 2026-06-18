# Driver Drowsiness Detection System

## Overview

Driver Drowsiness Detection System is a real-time AI-powered driver monitoring application that detects signs of fatigue through a webcam.

The system combines:

- Deep Learning (`EfficientNetB0_FastKAN`) for drowsiness classification
- MediaPipe Face Mesh for facial landmark detection
- Rule-based analysis using EAR (Eye Aspect Ratio), MAR (Mouth Aspect Ratio), and head pose estimation
- Real-time warning dashboard with audio alerts

> **Important**
>
> The pretrained model is not included in this repository due to GitHub file size limitations.
>
> Download:
>
> https://drive.google.com/file/d/1Jo0hj1ZQt9GZ7aDssd63ownw-j7147kq/view?usp=drive_link
>
> Place the file in:
>
> weights/best_EfficientNetB0_FastKAN.pth

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
Drowsiness-Detection-System/
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
git clone https://github.com/thuantcn1802/Drowsiness-Detection-System.git
cd Drowsiness-Detection-System
```

### 2. Create a virtual environment

#### Option 1: Conda (Recommended)

```bash
conda create -n drowsiness python=3.10 -y
conda activate drowsiness
```

#### Option 2: venv

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Obtain the Trained Model

The application requires the following model file:

```text
best_EfficientNetB0_FastKAN.pth
```

There are two ways to obtain it:

#### Option 1: Download the Pretrained Model (Recommended)

Download the pretrained model from Google Drive:

<https://drive.google.com/file/d/1Jo0hj1ZQt9GZ7aDssd63ownw-j7147kq/view?usp=drive_link>

Create the `weights` directory if it does not exist:

```bash
mkdir weights
```

Place the downloaded file in:

```text
Drowsiness-Detection-System/
└── weights/
    └── best_EfficientNetB0_FastKAN.pth
```

#### Option 2: Train the Model Yourself

The project includes training scripts inside the `train/` directory.

Prepare the dataset and run the training pipeline:

```bash
python train/train.py
```

After training is completed, copy the generated model file to:

```text
Drowsiness-Detection-System/
└── weights/
    └── best_EfficientNetB0_FastKAN.pth
```

> **Note**
>
> Training the model requires the corresponding dataset and may take a significant amount of time depending on your hardware configuration.

Without a valid model file in the `weights` directory, the application cannot perform drowsiness detection.

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
