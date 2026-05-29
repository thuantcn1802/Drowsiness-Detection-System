# Driver Drowsiness Detection System

## Giới thiệu

Hệ thống phát hiện buồn ngủ của tài xế sử dụng Computer Vision.

Chương trình sử dụng:

* MediaPipe Face Mesh để phát hiện các điểm mốc trên khuôn mặt.
* EAR (Eye Aspect Ratio) để xác định trạng thái mở/nhắm mắt.
* OpenCV để xử lý hình ảnh từ webcam.
* Pygame để phát cảnh báo âm thanh khi phát hiện buồn ngủ.

---

## Công nghệ sử dụng

* Python 3.10+
* OpenCV
* MediaPipe
* NumPy
* SciPy
* Pygame

---

## Cấu trúc thư mục

```text
driver-drowsiness-system/
│
├── drowsiness_ear.py
├── requirements.txt
├── alarm.wav
├── README.md
└── .gitignore
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

```bash
python drowsiness_ear.py
```

---

## Nguyên lý hoạt động

1. Webcam thu hình khuôn mặt người dùng.
2. MediaPipe Face Mesh xác định các điểm mốc của mắt.
3. Tính toán Eye Aspect Ratio (EAR).
4. Nếu EAR nhỏ hơn ngưỡng trong một khoảng thời gian liên tục:

   * Xác định người dùng đang buồn ngủ.
   * Hiển thị cảnh báo trên màn hình.
   * Phát âm thanh cảnh báo.

---

## Thành viên thực hiện

* Đặng Công Bằng

---

## Repository

GitHub:

https://github.com/Bang0409/driver-drowsiness-system
