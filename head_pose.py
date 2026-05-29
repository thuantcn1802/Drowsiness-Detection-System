def is_head_down(nose_y, eye_avg_y, threshold):
    return nose_y > eye_avg_y + threshold