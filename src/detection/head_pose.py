# head_pose.py
# Phát hiện gục/cúi đầu bằng điểm mốc MediaPipe Face Mesh.

import time


class HeadDownDetector:
    """Phát hiện gục đầu bằng cách so sánh vị trí mũi với vùng mắt.

    Cách làm:
    - Lấy score hiện tại = nose_y - eye_center_y.
    - Lấy trung bình N frame đầu làm tư thế đầu bình thường.
    - Nếu score tăng quá score_delta trong down_time giây thì báo HEAD DOWN.

    Dùng tọa độ normalized của MediaPipe nên ít phụ thuộc kích thước frame.
    """

    def __init__(self, calibration_frames=30, score_delta=0.025, down_time=1.5):
        self.calibration_frames = calibration_frames
        self.score_delta = score_delta
        self.down_time = down_time

        self.neutral_scores = []
        self.neutral_score = None
        self.down_start_time = None

    @property
    def is_calibrated(self):
        return self.neutral_score is not None

    def reset(self):
        self.neutral_scores = []
        self.neutral_score = None
        self.down_start_time = None

    def reset_timer(self):
        self.down_start_time = None

    def calculate_score(self, face_landmarks):
        # MediaPipe indices
        nose = face_landmarks.landmark[1]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]

        eye_center_y = (left_eye.y + right_eye.y) / 2.0
        score = nose.y - eye_center_y
        return float(score)

    def update(self, face_landmarks):
        """Trả về: (is_head_alert, head_score, head_delta, is_calibrated)."""
        current_score = self.calculate_score(face_landmarks)

        if self.neutral_score is None:
            self.neutral_scores.append(current_score)

            if len(self.neutral_scores) >= self.calibration_frames:
                self.neutral_score = sum(self.neutral_scores) / len(self.neutral_scores)
                print(f"Da calibration guc dau: neutral_score = {self.neutral_score:.3f}")

            return False, current_score, 0.0, False

        head_delta = current_score - self.neutral_score
        is_down_now = head_delta >= self.score_delta

        if is_down_now:
            if self.down_start_time is None:
                self.down_start_time = time.time()
        else:
            self.down_start_time = None

        if self.down_start_time is not None:
            duration = time.time() - self.down_start_time
            is_alert = duration >= self.down_time
        else:
            is_alert = False

        return is_alert, current_score, head_delta, True


def is_head_down(nose_y, eye_avg_y, threshold):
    """Hàm cũ để tương thích với code cũ nếu còn import.

    threshold là khoảng cách normalized giữa mũi và trung bình mắt.
    """
    return (nose_y - eye_avg_y) > threshold