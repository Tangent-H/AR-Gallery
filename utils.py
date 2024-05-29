import cv2
import numpy as np
import os
from PIL import Image
import glob

ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# def cover_aruco(corners, ids, rejected, image, bg_color):
# 	if len(corners) > 0:
# 		# flatten the ArUco IDs list
# 		ids = ids.flatten()
# 		# loop over the detected ArUCo corners
# 		for (markerCorner, markerID) in zip(corners, ids):
# 			# extract the marker corners (which are always returned in
# 			# top-left, top-right, bottom-right, and bottom-left order)
# 			corners = markerCorner.reshape((4, 2))
# 			(topLeft, topRight, bottomRight, bottomLeft) = corners
# 			# convert each of the (x, y)-coordinate pairs to integers
# 			topRight = (int(topRight[0]), int(topRight[1]))
# 			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
# 			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
# 			topLeft = (int(topLeft[0]), int(topLeft[1]))



# 			cv2.line(image, topLeft, topRight, bg_color, 5)
# 			cv2.line(image, topRight, bottomRight, bg_color, 5)
# 			cv2.line(image, bottomRight, bottomLeft, bg_color, 5)
# 			cv2.line(image, bottomLeft, topLeft, bg_color, 5)
# 			cv2.fillPoly(image, [corners.astype(np.int32).reshape((-1, 1, 2))], bg_color)
# 			# compute and draw the center (x, y)-coordinates of the ArUco
# 			# marker
# 			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
# 			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
# 			# cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
# 			# draw the ArUco marker ID on the image
# 			# cv2.putText(image, str(markerID),(topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
# 				# 0.5, (0, 255, 0), 2)
# 			print("[Inference] ArUco marker ID: {}".format(markerID))
# 			# show the output image
# 	return image

def cover_aruco(corners, ids, rejected, image, bg_color, aspect_ratios):
    adjusted_corners = []

    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()

        # loop over the detected ArUCo corners
        for (markerCorner, markerID, aspect_ratio) in zip(corners, ids, aspect_ratios):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            marker_corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = marker_corners

            # Compute the width and height of the ArUco marker
            aruco_width = np.linalg.norm(topRight - topLeft)
            aruco_height = aruco_width

            # Calculate the new height based on aspect ratio
            new_height = aruco_width * aspect_ratio

            # Calculate the new corners in the normalized space
            rect_points = np.array([
                [-aruco_width / 2, new_height / 2],
                [aruco_width / 2, new_height / 2],
                [aruco_width / 2, -new_height / 2],
                [-aruco_width / 2, -new_height / 2]
            ], dtype=np.float32)

            # Find homography from normalized space to marker corners
            h, _ = cv2.findHomography(np.array([
                [-aruco_width / 2, aruco_height / 2],
                [aruco_width / 2, aruco_height / 2],
                [aruco_width / 2, -aruco_height / 2],
                [-aruco_width / 2, -aruco_height / 2]
            ], dtype=np.float32), marker_corners)

            # Compute the new corners in image space
            new_corners = cv2.perspectiveTransform(rect_points.reshape(-1, 1, 2), h).reshape(-1, 2)

            # Ensure the coordinates are in tuple and integer form
            new_topLeft, new_topRight, new_bottomRight, new_bottomLeft = map(lambda pt: (int(pt[0]), int(pt[1])), new_corners)

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            cv2.line(image, topLeft, topRight, bg_color, 5)
            cv2.line(image, topRight, bottomRight, bg_color, 5)
            cv2.line(image, bottomRight, bottomLeft, bg_color, 5)
            cv2.line(image, bottomLeft, topLeft, bg_color, 5)
            cv2.fillPoly(image, [marker_corners.astype(np.int32).reshape((-1, 1, 2))], bg_color)
            # cv2.line(image, new_topLeft, new_topRight, bg_color, 5)
            # cv2.line(image, new_topRight, new_bottomRight, bg_color, 5)
            # cv2.line(image, new_bottomRight, new_bottomLeft, bg_color, 5)
            # cv2.line(image, new_bottomLeft, new_topLeft, bg_color, 5)
            # cv2.fillPoly(image, [np.array([new_topLeft, new_topRight, new_bottomRight, new_bottomLeft], dtype=np.int32).reshape((-1, 1, 2))], bg_color)

            print("[Inference] ArUco marker ID: {}".format(markerID))

            # add adjusted corners to the list
            adjusted_corners.append(np.array([new_topLeft, new_topRight, new_bottomRight, new_bottomLeft], dtype=np.float32))

    return image, adjusted_corners


def get_image_aspect_ratios(folder_path):
    # 获取指定文件夹下所有.jpg文件
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    aspect_ratios = []

    for image_file in image_files:
        with Image.open(image_file) as img:
            width, height = img.size
            aspect_ratio =  height / width
            aspect_ratios.append(aspect_ratio)

    return image_files, aspect_ratios