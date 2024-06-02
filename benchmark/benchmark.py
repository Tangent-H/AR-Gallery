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
ap.add_argument("-i", "--camera", type=str,default="False",required=False, help="Set to True if using webcam")
ap.add_argument("-v", "--video", type=str, default= "test.mp4", help="Path to the video file")
ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
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
error_cnt = 0
frame_cnt = 0
num_better = 0
num_worse = 0
while True:
	ret, frame = video.read()
	
	if ret is False:
		break


	h, w, _ = frame.shape

	frame_cnt += 1

	width=640
	height = int(width*(h/w))

	

	frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
	frame_copy = frame.copy()

	start_time_buf.append(time.time())
	corners, ids, rejected =detector.detectMarkers(frame)
	corners1, ids1, rejected1 = detect_aruco.detect_markers_wrapper(frame_copy)
	stop_time_buf.append(time.time())


	detected_markers = aruco_display(corners, ids, rejected, frame)
	detected_markers1 = aruco_display(corners1,ids1, rejected1, frame_copy)
	
	if len(corners) != len(corners1):
		print(f"Detected markers count mismatch: {len(corners)} != {len(corners1)}")
		error_cnt += 1
		if len(corners) < len(corners1):
			num_better += 1
		elif len(corners) > len(corners1):
			num_worse += 1
		cv2.imwrite(f"results/cv_{frame_cnt}.jpg",detected_markers)
		cv2.imwrite(f"results/ours_{frame_cnt}.jpg",detected_markers1)
		# cv2.imshow("Image", detected_markers)
		# cv2.imshow("Image1", detected_markers1)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

print(f"Error count: {error_cnt}")
print(f"Average FPS: {1/np.mean(np.array(stop_time_buf) - np.array(start_time_buf))}")
print(f"Width: {width}, Height: {height}")
print(f"Number of frames: {frame_cnt}")
print(f"Number of better: {num_better}")
print(f"Number of worse: {num_worse}")
cv2.destroyAllWindows()
video.release()