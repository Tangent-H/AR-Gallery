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

If you want to benchmark the result of our method and OpenCV’s method, go to `benchmark` folder, run

```
python benchmark.py
```

And you can find the difference of failure cases between our method and OpenCV’s method.

Testing the average FPS needs to do a little bit modification to `benchmark.py`. The main change lies in line 66-69. You need to comment out one of the detection method and reserve the other one.

```python
	start_time_buf.append(time.time())
	corners, ids, rejected =detector.detectMarkers(frame)	# Opencv method
	corners1, ids1, rejected1 = detect_aruco.detect_markers_wrapper(frame_copy) # our method
	stop_time_buf.append(time.time())
```

Also, you can alternate these two methods in 141-142 line code in `TransFussion_poisson.py` to check the different effect by direct fussion and possion fussion:

	# nomal_clone = cv2.seamlessClone(obj_croped, background, mask, center, cv2.NORMAL_CLONE) # direct fussion
	#nomal_clone = cv2.seamlessClone(obj_croped,  background, mask, center, cv2.MIXED_CLONE) # possion fussion

## Test of Image Fussion

The Poisson Fussion is implement in `poissonFussion_manual.py`, you can test it directly by running 

	python poissonFussion_manual.py

It will try to fuse `./test/fussion_plane.jpg` with `./test/fussion_underwater.jpg`. You can also change mask to alternate the fussion part of background figure by changing mask:

	mask = np.array([[400, 400], [300, 200]]) # [center, window_size] 

You can also try fuse another two figures. Just fill the filepath on last code of `poissonFussion_manual.py`:

	if __name__ == "__main__":
		main('foreground_filtpath', 'background_filepath')

Because the manual implementation of Poisson Fussion is too slow, we recommend you to use `TransFussion_poisson.py` to do video figure fussion, which uses the library of **OpenCV**, instead of `poissonFussion_manual.py`. `poissonFussion_manual.py` is only used for two single figures' fussion. 



