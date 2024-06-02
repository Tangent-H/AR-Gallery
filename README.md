# ART: Augmented Reality Tapestry

This paper presents an implementation of an Augmented Reality (AR) system that enhances gallery experiences through the integration of ArUco marker detection and Poisson blending techniques. The system leverages the robustness of ArUco for accurate marker recognition and employs Poisson blending to achieve seamless image integration. The combination of these methods allows for the creation of a visually coherent AR tapestry, contributing to a more immersive and interactive gallery environment.

## Usage

To run the script and have a view of our demo, enter the root directory of this repository and run:

```
python detect_aruco_video.py --type DICT_5X5_100 --camera True
```

The `--camera` option opens the default camera on your computer.

If your computer do not have a camera, use 

```
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video Video/test.mp4
```

Note that you need to prepare a `test.mp4` which contains ArUco markers in the DICT_5X5_100 under the `Video` folder. Our implementation only support detecting ArUco with id 1to 5 in this DICT.

If you want to compare the ArUco detection result using the OpenCV ArUco library with our method, go to line 59-60 of file `detect_aruco_video.py`. You can switch between these two methods by commenting out the other one.

```python
	# corners, ids, rejected =detector.detectMarkers(frame) # opencv
	corners, ids, rejected = detect_aruco.detect_markers_wrapper(frame) # ours
```

If you want to benchmark the result of our method and OpenCV’s method, go to benchmark folder, run

```
python benmark.py
```

And you can find
