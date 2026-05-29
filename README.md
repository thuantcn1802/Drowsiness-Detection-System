# Driver Drowsiness Detection System

## Giới thiệu

Driver Drowsiness Detection System là hệ thống phát hiện buồn ngủ của tài xế theo thời gian thực bằng Computer Vision và MediaPipe Face Mesh.

Hệ thống sử dụng webcam để theo dõi khuôn mặt người dùng và phân tích các dấu hiệu mất tập trung hoặc buồn ngủ thông qua:

* Eye Aspect Ratio (EAR) – phát hiện nhắm mắt kéo dài.
* Yawn Detection – phát hiện ngáp.
* Head Pose Estimation – phát hiện cúi đầu hoặc lệch đầu bất thường.
* Cảnh báo âm thanh và giao diện trực quan khi phát hiện nguy cơ buồn ngủ.

---

## Tính năng chính

* Phát hiện mắt nhắm bằng EAR.
* Phát hiện ngáp bằng khoảng cách môi.
* Ước lượng tư thế đầu (Head Pose).
* Hiển thị trạng thái thời gian thực trên giao diện.
* Phát cảnh báo âm thanh khi phát hiện buồn ngủ.
* Hoạt động trực tiếp từ webcam.

---

## Công nghệ sử dụng

* Python 3.10+
* OpenCV
* MediaPipe Face Mesh
* NumPy
* SciPy
* Pygame
* Tkinter

---

## Cấu trúc dự án

```text
driver-drowsiness-system/
│
├── main.py                 # Chương trình chính
├── gui.py                  # Giao diện người dùng
├── gui_camera.py           # Hiển thị camera trên GUI
│
├── eye.py                  # Tính EAR và phát hiện nhắm mắt
├── yawn.py                 # Phát hiện ngáp
├── head_pose.py            # Ước lượng tư thế đầu
├── drowsiness_ear.py       # Module phát hiện buồn ngủ bằng EAR
│
├── sounds/
│   └── alarm.mp3           # Âm thanh cảnh báo
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Kiến trúc hệ thống

```text
Webcam
   │
   ▼
MediaPipe Face Mesh
   │
   ├────────► Eye Detection (EAR)
   │
   ├────────► Yawn Detection
   │
   └────────► Head Pose Estimation
                │
                ▼
        Drowsiness Decision
                │
        ┌───────┴────────┐
        ▼                ▼
      GUI          Alarm Sound
```

---

## Cài đặt

### 1. Clone project

```bash
git clone https://github.com/Bang0409/driver-drowsiness-system.git
cd driver-drowsiness-system
```

### 2. Tạo môi trường ảo

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## Chạy chương trình

Chạy chương trình chính:

```bash
python gui_camera.py
```

---

## Nguyên lý hoạt động

### Bước 1: Thu nhận hình ảnh

Webcam liên tục thu hình khuôn mặt người dùng.

### Bước 2: Trích xuất landmark khuôn mặt

MediaPipe Face Mesh xác định các điểm mốc trên mắt, môi và khuôn mặt.

### Bước 3: Phân tích trạng thái

* Tính EAR để xác định mắt mở hay nhắm.
* Tính khoảng cách môi để phát hiện ngáp.
* Tính góc quay đầu để xác định tư thế đầu.

### Bước 4: Đưa ra quyết định

Nếu người dùng có dấu hiệu:

* Nhắm mắt kéo dài.
* Ngáp liên tục.
* Cúi đầu bất thường.

Hệ thống sẽ xác định trạng thái buồn ngủ.

### Bước 5: Cảnh báo

* Hiển thị cảnh báo trên giao diện.
* Phát âm thanh cảnh báo để đánh thức người lái.

---

## Kết quả mong đợi

* Phát hiện chính xác trạng thái buồn ngủ.
* Hoạt động thời gian thực.
* Hỗ trợ nâng cao an toàn khi lái xe.

---

## Thành viên thực hiện

* Đặng Công Bằng

---

## GitHub Repository

https://github.com/Bang0409/driver-drowsiness-system
