# Camera
CAMERA_INDEX = 0

# Rule-based thresholds

# Nhắm mắt
EAR_THRESHOLD = 0.22
CLOSED_EYES_TIME = 2.0

# Ngáp rule-based
MAR_THRESHOLD = 0.45


# Alarm
ENABLE_ALARM_SOUND = True
ALARM_SOUND_PATH = "assets/sounds/alarm.mp3"


# Gục / cúi đầu
HEAD_CALIBRATION_FRAMES = 30
HEAD_DOWN_SCORE_DELTA = 0.025
HEAD_DOWN_TIME = 1.5


# Deep learning model config
ENABLE_CNN_MODEL = True
MODEL_TYPE = "EfficientNetB0_FastKAN"
CNN_MODEL_PATH = "weights/best_EfficientNetB0_FastKAN.pth"
CNN_IMG_SIZE = 224
USE_IMAGENET_NORMALIZE = True

# Ngưỡng confidence chung cho CNN
CNN_CONFIDENCE_THRESHOLD = 0.70

# Lịch sử dự đoán CNN
CNN_HISTORY_SIZE = 7
CNN_DROWSY_THRESHOLD = 5
CNN_PREDICT_EVERY_N_FRAMES = 3


# CNN classes
CNN_CLASS_NAMES = [
    "notdrowsy",
    "sleepyCombination",
    "slowBlinkWithNodding",
    "yawning",
]

# Không để "yawning" ở đây nữa vì yawning dễ báo nhầm.
# Yawning sẽ được xác nhận riêng bằng confidence + MAR.
CNN_DROWSY_CLASSES = [
    "sleepyCombination",
    "slowBlinkWithNodding",
]


# Xác nhận riêng cho AI yawning
AI_YAWNING_CONF_THRESHOLD = 0.85
AI_YAWNING_MAR_THRESHOLD = 0.50


# GUI display config
APP_WIDTH = 1220
APP_HEIGHT = 720

CAMERA_DISPLAY_WIDTH = 1000
CAMERA_DISPLAY_HEIGHT = 615


# GUI text display config
CNN_TEXT_SCALE = 0.65
CNN_TEXT_THICKNESS = 2

CNN_PROB_TEXT_SCALE = 0.45
CNN_PROB_TEXT_THICKNESS = 1

CNN_TEXT_X = 30
CNN_TEXT_Y = 45
CNN_PROB_START_Y = 75
CNN_PROB_LINE_GAP = 22


# Debug CNN
ENABLE_CNN_DEBUG = True
CNN_DEBUG_SAVE_IMAGE = True
CNN_DEBUG_PRINT_PROBS = True
CNN_DEBUG_DIR = "debug/debug_cnn"
CNN_DEBUG_SAVE_EVERY_N = 10

SHOW_CAMERA_DEBUG_TEXT = False