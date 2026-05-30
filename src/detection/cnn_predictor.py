# cnn_predictor.py
# Load EfficientNetB0_FastKAN và dự đoán trạng thái từ frame webcam.

from collections import deque
from pathlib import Path
import time

import cv2
import torch
from PIL import Image
from torchvision import transforms

from src.config.config import (
    CNN_CLASS_NAMES,
    CNN_CONFIDENCE_THRESHOLD,
    CNN_DROWSY_CLASSES,
    CNN_DROWSY_THRESHOLD,
    CNN_HISTORY_SIZE,
    CNN_IMG_SIZE,
    CNN_MODEL_PATH,
    USE_IMAGENET_NORMALIZE,
)

# Import debug config an toàn, nếu config.py chưa có thì dùng mặc định.
try:
    from src.config.config import (
        ENABLE_CNN_DEBUG,
        CNN_DEBUG_SAVE_IMAGE,
        CNN_DEBUG_PRINT_PROBS,
        CNN_DEBUG_DIR,
        CNN_DEBUG_SAVE_EVERY_N,
    )
except Exception:
    ENABLE_CNN_DEBUG = False
    CNN_DEBUG_SAVE_IMAGE = False
    CNN_DEBUG_PRINT_PROBS = False
    CNN_DEBUG_DIR = "debug/debug_cnn"
    CNN_DEBUG_SAVE_EVERY_N = 10

from src.models.model import EfficientNetB0_FastKAN


class CNNPredictor:
    def __init__(self, model_path=CNN_MODEL_PATH, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_path = Path(model_path)
        self.class_names = CNN_CLASS_NAMES
        self.drowsy_classes = set(CNN_DROWSY_CLASSES)
        self.history = deque(maxlen=CNN_HISTORY_SIZE)
        self.predict_count = 0

        self.debug_dir = Path(CNN_DEBUG_DIR)
        if ENABLE_CNN_DEBUG and CNN_DEBUG_SAVE_IMAGE:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.model = EfficientNetB0_FastKAN(
            num_classes=len(self.class_names),
            pretrained=False,
        ).to(self.device)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay model: {self.model_path}. "
                f"Hay dat best_EfficientNetB0_FastKAN.pth cung thu muc project."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        transform_list = [
            transforms.Resize((CNN_IMG_SIZE, CNN_IMG_SIZE)),
            transforms.ToTensor(),
        ]

        if USE_IMAGENET_NORMALIZE:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                )
            )

        self.transform = transforms.Compose(transform_list)

        print("CNN Predictor loaded")
        print(f"Device: {self.device}")
        print(f"Model path: {self.model_path}")
        print(f"Class names: {self.class_names}")
        print(f"Drowsy classes for history: {self.drowsy_classes}")
        print(f"Use ImageNet normalize: {USE_IMAGENET_NORMALIZE}")

    def crop_face_from_landmarks(self, frame_bgr, face_landmarks, margin=0.40):
        """Crop khuôn mặt từ MediaPipe landmarks.

        margin=0.40 giúp crop rộng hơn để model thấy cả mắt, miệng, cằm.
        """
        h, w, _ = frame_bgr.shape

        xs = [lm.x for lm in face_landmarks.landmark]
        ys = [lm.y for lm in face_landmarks.landmark]

        x1 = int(max(min(xs) * w, 0))
        y1 = int(max(min(ys) * h, 0))
        x2 = int(min(max(xs) * w, w - 1))
        y2 = int(min(max(ys) * h, h - 1))

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 0 or box_h <= 0:
            return frame_bgr

        pad_x = int(box_w * margin)
        pad_y = int(box_h * margin)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        if x2 <= x1 or y2 <= y1:
            return frame_bgr

        return frame_bgr[y1:y2, x1:x2]

    def save_debug_image(self, crop_bgr, label, confidence):
        """Lưu ảnh crop mà model đang dùng để dự đoán."""
        if not (ENABLE_CNN_DEBUG and CNN_DEBUG_SAVE_IMAGE):
            return

        if self.predict_count % CNN_DEBUG_SAVE_EVERY_N != 0:
            return

        timestamp = int(time.time() * 1000)
        filename = f"{self.predict_count:06d}_{label}_{confidence:.2f}_{timestamp}.jpg"
        save_path = self.debug_dir / filename
        cv2.imwrite(str(save_path), crop_bgr)

    def print_debug_probs(self, label, confidence, probs, alert, drowsy_count):
        """In xác suất từng class ra terminal."""
        if not (ENABLE_CNN_DEBUG and CNN_DEBUG_PRINT_PROBS):
            return

        print("\n========== CNN DEBUG ==========")
        print(f"Predict #{self.predict_count}")
        print(f"Top label: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"History: {drowsy_count}/{self.history.maxlen}")
        print(f"Alert: {alert}")
        print("Probabilities:")

        for name, prob in zip(self.class_names, probs):
            print(f"  {name}: {float(prob):.4f}")

        print("================================\n")

    def predict(self, frame_bgr, face_landmarks=None):
        self.predict_count += 1

        if face_landmarks is not None:
            crop_bgr = self.crop_face_from_landmarks(frame_bgr, face_landmarks)
        else:
            crop_bgr = frame_bgr

        image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probs_tensor = torch.softmax(logits, dim=1)[0]
            confidence, pred_idx = torch.max(probs_tensor, dim=0)

        label = self.class_names[pred_idx.item()]
        conf = float(confidence.item())
        probs = probs_tensor.detach().cpu().numpy()

        is_drowsy = label in self.drowsy_classes and conf >= CNN_CONFIDENCE_THRESHOLD
        self.history.append(1 if is_drowsy else 0)

        drowsy_count = sum(self.history)
        alert = len(self.history) == self.history.maxlen and drowsy_count >= CNN_DROWSY_THRESHOLD

        self.save_debug_image(crop_bgr, label, conf)
        self.print_debug_probs(label, conf, probs, alert, drowsy_count)

        return {
            "label": label,
            "confidence": conf,
            "is_drowsy": is_drowsy,
            "drowsy_count": drowsy_count,
            "history_size": self.history.maxlen,
            "alert": alert,
            "probs": probs,
        }