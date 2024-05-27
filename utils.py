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
    new_corners = np.zeros(1)
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for i, (markerCorner, markerID) in enumerate(zip(corners, ids)):
            # extract the marker corners (which are always returned in
            # top-left, top-right, bottom-right, and bottom-left order)
            markerCorners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = markerCorners
            
            # convert each of the (x, y)-coordinate pairs to integers
            topLeft = (int(topLeft[0]), int(topLeft[1]))
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

            # 获取相应的宽高比
            aspect_ratio = aspect_ratios[i] if i < len(aspect_ratios) else 1.0

            # 计算新的corners
            if aspect_ratio > 1:
                width = np.linalg.norm(np.array(topRight) - np.array(topLeft))
                height = np.linalg.norm(np.array(topLeft) - np.array(bottomLeft))
                new_width = width * aspect_ratio
                delta_width = (new_width - width) / 2
                
                new_topLeft = (topLeft[0] - delta_width, topLeft[1])
                new_topRight = (topRight[0] + delta_width, topRight[1])
                new_bottomLeft = (bottomLeft[0] - delta_width, bottomLeft[1])
                new_bottomRight = (bottomRight[0] + delta_width, bottomRight[1])

                new_corners = np.array([new_topLeft, new_topRight, new_bottomRight, new_bottomLeft], dtype=np.int32)
            else:
                new_corners = markerCorners.astype(np.int32)

            # 绘制图像
            # bg_color = (0,255,0)
            cv2.line(image, tuple(new_corners[0]), tuple(new_corners[1]), bg_color, 5)
            cv2.line(image, tuple(new_corners[1]), tuple(new_corners[2]), bg_color, 5)
            cv2.line(image, tuple(new_corners[2]), tuple(new_corners[3]), bg_color, 5)
            cv2.line(image, tuple(new_corners[3]), tuple(new_corners[0]), bg_color, 5)
            cv2.fillPoly(image, [new_corners.reshape((-1, 1, 2))], bg_color)

            # 计算和绘制中心点
            cX = int((new_corners[0][0] + new_corners[2][0]) / 2.0)
            cY = int((new_corners[0][1] + new_corners[2][1]) / 2.0)
            # cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
            # cv2.putText(image, str(markerID), (new_corners[0][0], new_corners[0][1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # print("[Inference] ArUco marker ID: {}".format(markerID))

    return image,new_corners

def get_image_aspect_ratios(folder_path):
    # 获取指定文件夹下所有.jpg文件
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    aspect_ratios = []

    for image_file in image_files:
        with Image.open(image_file) as img:
            width, height = img.size
            aspect_ratio = width / height
            aspect_ratios.append(aspect_ratio)

    return aspect_ratios