'''
python benchmark.py --type DICT_5X5_100 --camera False --video test.mp4
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
import detect_aruco

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

start_time_buf = []
stop_time_buf = []

while True:
	ret, frame = video.read()
	
	if ret is False:
		break


	h, w, _ = frame.shape

	width=640
	height = int(width*(h/w))
	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)

	start_time_buf.append(time.time())
	# corners, ids, rejected =detector.detectMarkers(frame)
	corners, ids, rejected = detect_aruco.detect_markers_wrapper(frame)
	stop_time_buf.append(time.time())


	detected_markers = aruco_display(corners, ids, rejected, frame)
	

	cv2.imshow("Image", detected_markers)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break


print(f"Average FPS: {1/np.mean(np.array(stop_time_buf) - np.array(start_time_buf))}")

cv2.destroyAllWindows()
video.release()