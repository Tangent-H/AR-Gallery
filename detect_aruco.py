import cv2
import numpy as np
from cv2 import aruco

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def detect_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_corners(contours):
    corner_candidates = []
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            corner_candidates.append(approx)
    return corner_candidates

def verify_and_decode_markers(image, aruco_dict, detector_params):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    detected_markers = []
    if ids is not None:
        for i, corner in enumerate(corners):
            marker_id = ids[i][0]
            detected_markers.append((marker_id, corner))
    return detected_markers

def main(image_path, aruco_dict, detector_params):
    image = cv2.imread(image_path)
    binary_image = preprocess_image(image)
    contours = detect_contours(binary_image)
    corners = detect_corners(contours)
    detected_markers = verify_and_decode_markers(image, aruco_dict, detector_params)

    for marker_id, corner in detected_markers:
        pts = corner.reshape(4, 2).astype(int)
        for i in range(4):
            cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(image, str(marker_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detected ArUco Markers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "aruco0.png"
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    main(image_path, aruco_dict, detector_params)
