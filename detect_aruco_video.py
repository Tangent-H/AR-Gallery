'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video Video/test.mp4
python detect_aruco_video.py -i True -t DICT_5X5_100
'''

import numpy as np
from utils import ARUCO_DICT, cover_aruco
import argparse
import time
import cv2
import sys
from TransFussion import TransFussion, main_color_detect
import glob
import os


pipeline = "rtsp://10.32.90.53:18464/h264_ulaw.sdp"
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", required=True, help="Set to True if using webcam")
ap.add_argument("-v", "--video", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())

if args["camera"].lower() == "true":
	video = cv2.VideoCapture(0)
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

while True:
	ret, frame = video.read(0)
	
	if ret is False:
		break


	h, w, _ = frame.shape

	# width=250
	# height = int(width*(h/w))
	# frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	corners, ids, rejected =detector.detectMarkers(frame)
	bg_color = tuple(val.item() for val in main_color_detect(frame).flatten())
	_  = cover_aruco(corners, ids, rejected, frame, bg_color)
	pics = glob.glob(os.path.join("Gallery", "*.jpg"))
	render = np.zeros(1)
	if corners and len(corners) > 0:
		i = 0
		for corner in corners:
			i += 1
			corner = corner.reshape(4,2).astype(np.uint32).tolist()
			# corners = corners[0].reshape(4,2).astype(np.uint32).tolist()
			if i == 1:
				render = TransFussion(frame, pics[i], corner, 1)
			else:
				render = TransFussion(render, pics[i], corner, 1)
	# corners = [int(corner) for corner in corners]
	
	if render.any():
		cv2.imshow("Image", render)
	else:
		cv2.imshow("Image", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()