from src.detection.eye import euclidean_distance

def calculate_mar(mouth_points):
    p1, p2, p3, p4, p5, p6 = mouth_points

    vertical = euclidean_distance(p1, p2)
    horizontal = euclidean_distance(p3, p4)

    return vertical / horizontal