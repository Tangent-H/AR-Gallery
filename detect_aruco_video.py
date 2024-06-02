'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video Video/test.mp4
python detect_aruco_video.py -i True -t DICT_5X5_100
'''

import numpy as np
from utils import *
import argparse
import time
import cv2
import sys
from TransFussion import TransFussion, main_color_detect
import glob
import os
import detect_aruco


pipeline = "http://10.24.165.210:4747/video"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
	video = cv2.VideoCapture(pipeline)
	time.sleep(2.0)
	
else:
	if args["video"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

	video = cv2.VideoCapture(args["video"])

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams) 
pics, aspect_ratio = get_image_aspect_ratios("Gallery")
aspect_ratio = np.array(aspect_ratio)
while True:
	ret, frame = video.read(0)
	
	if ret is False:
		break


	h, w, _ = frame.shape

	width=640
	height = int(width*(h/w))
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	# corners, ids, rejected =detector.detectMarkers(frame)
	corners, ids, rejected = detect_aruco.detect_markers_wrapper(frame)
	bg_color = tuple(val.item() for val in main_color_detect(frame).flatten())
	render = np.zeros(1)
	if ids is not None and ids.any() and np.all((ids >= 1) & (ids <= 5)):
		ids_copy = [int(id) for id in (ids-1).flatten().tolist()]
		print(f"ids: {ids_copy}")
		pics_show = []
		aspect_ratio_show = []
		for id in ids_copy:
			pics_show.append(pics[id])
			aspect_ratio_show.append(aspect_ratio[id])
		print(f"pics_show: {pics_show}")
		print(f"pics: {pics}; aspect_ratio: {aspect_ratio}")
		_,adjusted_corners = cover_aruco(corners, ids, rejected, frame, bg_color, aspect_ratio_show)
		if  adjusted_corners is not None and len(adjusted_corners) > 0:
			i = 0
			for id in ids:
				id = int(np.clip(id, 1, 5))
				corner = adjusted_corners[i]
				corner = corner.reshape(4,2).astype(np.uint32).tolist()
				if i == 0:
					render = TransFussion(frame, pics_show[i], corner, 1.5)
				else:
					render = TransFussion(render, pics_show[i], corner, 1.5)
				i += 1
		# corners = [int(corner) for corner in corners]
	
	if render is not None and render.any():
		cv2.imshow("Image", render)
	else:
		cv2.imshow("Image", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()