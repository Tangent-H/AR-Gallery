import cv2
import numpy as np


ARUCO_DICT = {"1":np.array([[0,0,0,0,1],
                            [1,1,0,0,0],
                            [0,0,0,0,1],
                            [1,0,1,1,1],
                            [0,0,1,1,0]]),
                "2":np.array([[1,1,0,1,0],
                            [1,1,1,1,0],
                            [0,0,0,1,1],
                            [1,0,1,1,0],
                            [1,1,1,0,1]]),
                "3":np.array([[1,0,0,0,0],
                            [0,0,1,1,1],
                            [0,0,1,0,1],
                            [0,1,1,1,1],
                            [1,0,1,1,1]]),
                "4":np.array([[1,1,0,1,0],
                            [1,1,1,0,1],
                            [0,1,1,0,1],
                            [0,1,0,0,1],
                            [0,0,1,0,0]]),
                "5":np.array([[1,1,1,0,1],
                            [0,1,0,0,0],
                            [0,0,0,1,0],
                            [0,0,0,0,1],
                            [0,1,1,0,1]]),
                            }
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    # _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_OTSU)
    
    return binary

def detect_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def detect_corners(contours):
    corner_candidates = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 600:  # 排除小面积区域
            corner_candidates.append(approx)
    return corner_candidates

def order_points(pts):
    center = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_pts = pts[sorted_indices]
    sorted_pts = sorted_pts.astype(np.float32)
    
    return sorted_pts

def extract_marker_id(corner_points, image):
    ordered_corners = order_points(corner_points.reshape(4, 2))
    # corner = ordered_corners.astype(np.int32)
    # cv2.circle(image, tuple(corner[0]), 5, (255, 255, 255), -1)
    # cv2.circle(image, tuple(corner[1]), 5, (0, 0, 255), -1)
    # cv2.circle(image, tuple(corner[2]), 5, (0, 255, 0), -1)
    # cv2.circle(image, tuple(corner[3]), 5, (255, 0, 0), -1)
    # cv2.imshow("Order corners", image)
    ordered_corners_np = np.array(ordered_corners, dtype=np.int32)
    # print(f"corner_points: {corner_points}, ordered_corners: {ordered_corners}")
    (tl, tr, br, bl) = ordered_corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    maxLength = max(maxWidth, maxHeight)

    dst = np.array([
        [0, 0],
        [maxLength - 1, 0],
        [maxLength - 1, maxLength - 1],
        [0,  maxLength- 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(ordered_corners, dst)
    # print(f"corners:{corner_points}, ordered_corners:{ordered_corners}, dst: {dst}")
    warped = cv2.warpPerspective(image, M, (maxLength,maxLength))
    # cv2.imshow("image", image)
    # cv2.imshow("warped", warped)

    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, warped_binary = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("warped_binary", warped_binary)
    # print(f"maxlength:{maxLength}")
    warped_binary = warped_binary[int(maxLength/7):int(maxLength/7*6),int(maxLength/7):int(maxLength/7*6)]
    aruco_binary = cv2.resize(warped_binary, (5, 5)) # for DICT_5X5
    aruco_binary = (aruco_binary > 0).astype(int)

    marker_id, aruco_binary_rot, rot = find_matching_aruco(aruco_binary)
    reagrranged_corners = ordered_corners_np
    if not marker_id == -1:
        # print(f"rot : {rot}")
        reagrranged_corners = np.roll(ordered_corners_np, rot, axis=0)
        # print(f"aruco_binary: {aruco_binary_rot}")
        # cv2.circle(image, reagrranged_corners[0],radius=5, color=(0, 255, 0), thickness=-1)
    return marker_id, reagrranged_corners

def find_matching_aruco(aruco_binary):
    """查找与aruco_binary匹配的ARUCO_DICT中的ID和旋转次数"""
    for rotation in range(4):
        for key, value in ARUCO_DICT.items():
            if np.array_equal(aruco_binary, value):
                return key, aruco_binary, rotation
        aruco_binary = np.rot90(aruco_binary, -1)
    return -1, aruco_binary, None

def verify_and_decode_markers(image):
    binary_image = preprocess_image(image)
    # cv2.imshow("Binary Image", binary_image)
    contours = detect_contours(binary_image)
    # visualize = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(visualize, contours, -1, (0,255,0), 2)
    # cv2.imshow("Binary Image", visualize)
    # cv2.imshow("Contours", image)
    corners = detect_corners(contours)

    detected_markers = []

    for corner in corners:
        # cv2.circle(image, tuple(corner[0][0]), 5, (255, 255, 255), -1)
        # cv2.circle(image, tuple(corner[1][0]), 5, (0, 0, 255), -1)
        # cv2.circle(image, tuple(corner[2][0]), 5, (0, 255, 0), -1)
        # cv2.circle(image, tuple(corner[3][0]), 5, (255, 0, 0), -1)

        marker_id, rearranged_corners = extract_marker_id(corner, image)
        detected_markers.append((marker_id, rearranged_corners))
        # cv2.imshow("Detected ArUco Markers", image)
    return detected_markers


def detect_markers_wrapper(frame):
    detected_markers = verify_and_decode_markers(frame)
    rejected = None
    ids = []
    corners = []
    for key, value in detected_markers:
        if key != -1:
            ids.append(key)
            corners.append(value)
    ids = np.array(ids,dtype=np.int32)
    return corners, ids, rejected

def main(image_path):
    image = cv2.imread(image_path)
    aspect_ratio = image.shape[0] / image.shape[1]
    image = cv2.resize(image, (640, int(640*aspect_ratio)))
    detected_markers = verify_and_decode_markers(image)

    for marker_id, corner in detected_markers:
        pts = corner.reshape(4, 2).astype(int)
        for i in range(4):
            cv2.line(image, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        cv2.putText(image, str(marker_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Detected ArUco Markers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = "test/aruco_test.jpg"
    main(image_path)
