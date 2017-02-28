##Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./output_images/camera_cal.png "Undistorted chessboard"
[image2]: ./output_images/test3.png "Good Threshold Example"
[image3]: ./output_images/test6.png "Problem Threshold Example"
[image4]: ./output_images/test_warped3.png "Good Warp Example"
[image5]: ./output_images/test_warped6.png "Problem Warp Example"
[image6]: ./output_images/test_final3.png "Good Output Example"
[image7]: ./output_images/test_final6.png "Problem Output Example"
[video1]: ./project_video_final.mp4 "Final Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibrate_camera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  

I detected chessboard corners on 20 images of the same chessboard taken from different angles using the `cv2.findChessboardCorners()` function. The set of these detected corners is stored in the `imgpoints` array.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction one of the calibration images using the `cv2.undistort()` function and obtained this result: 

![Figure 1][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like that in the top left panel of this example:
![Figure 2][image2]

The undistorted image can be seen in the top right. During camera calibration, I pickled the two transformations necessary to undistort images taken with the subject camera. In my pipeline, I load these transformations (`test_pipeline.py`, lines 186-190) and use the `cv2.undistort()` function to remove lens distortion (`test_pipeline.py`, line 205)

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color channel and HLS channel thresholds to generate a binary image. Using guidance from the lectures and some experimentation, I found that using the red channel along with a horizontal Sobel filter could detect lane lines reasonably well (`test_pipeline.py`, lines 15-22). To supplement this, I also used the saturation channel from the HLS representation of that image (`test_pipeline.py`, lines 25-26). The thresholds for these two color transforms were the following (`test_pipeline.py`, line 7).

| Red           | Saturation    | 
|:-------------:|:-------------:| 
| 40, 100       | 170, 255      | 

The layered output of these two operations can be seen above in the bottom left panel. The filtered and thresholded red channel is in red, while the thresholded saturation channel is in green.

The binary image created from the union of these two transformations (`test_pipeline.py`, line 34) can be seen in the lower right panel above.

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in a function called `create_perspective_transforms()`, which appears in lines 38 through 52 in the file `test_pipeline.py`.  The `create_perspective_transforms()` function takes as an input a tuple describing the transformed image size (`image_size`).  I chose to hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[575, 450],
    [725, 450],
    [1250, 700],
    [30, 700]])
dst = np.float32(
    [[0, 0],
    [img_size[0], 0],
    [img_size[0], img_size[1]],
    [0, img_size[1]]])

```

The input image was assumed to be 1280x720 pixels in size, hence the fixed dimensions on the first transformation. The output image was 600x1200 pixels (`test_pipeline.py`, line 193). The `src` mask describes a trapezoid, with the first point being in the top left and continuing clockwise from there. This is transformed to a rectangle in `dst`.

The transform created by `create_perspective_transforms()` is used at line 230 in `test_pipeline.py` to warp the lane area of the binary pipeline image into a rectangle with parallel lane lines. This can be seen in the first panel below.

![Figure 3][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the convolution method to initially find lane-line pixels (`test_pipeline.py`, lines 232-242). This method slides a window across the image, calculating a convolution at each place. This convolution is maximized when crossing hot pixels in the binary image. The method `find_window_centroids()` at line 87 in `test_pipeline.py` implements this approach. It accepts parameters describing the convolution kernel as well as a threshold of the maximum convolution value to use for detecting a lane marking. I found that it was necessary to not detect lane markings in some of the horizontal slices I scanned as otherwise they would drift. A convolution threshold with a formula in line 95 of `test_pipeline.py` proved useful.

To initialize the search, I considered the bottom quarter of the image. From there I divided the image into horizontal slices and passed the kernel of the left and right half of each slice, using the previously detected lane marking center as a seed for the next slice as I worked from the bottom of the image to the top. Detected lane windows, including skipped slices, can be seen in the second panel in the image above.

After detecting probable areas for the left and right lane lines, I selected only those hot pixels in the search areas and fit a second-order polynomial to each of the two lane lines. This process can be seen in the third panel above, and can be found in code in lines 284-292 of `test_pipeline.py`.

In the real pipeline, the fitted lane lines were used as the search area for the next image's lane lines. This was implemented in the `find_next_lane_line()` function (`test_pipeline.py` lines 145-166), which accepts the warped image, the previous left and right fits, and two optional scaling variables for transforming the fit to another coordinate space such as the real world. These latter scalars would be used when calculating the radius of curvature of the road.

The search area for this emthod applied to the same warped image can be seen in the fourth panel of the figure above, along with improved fits.

In the final pipeline, I would compare the new fits to an average of several previous good fits (`pipeline.py` line 270). If the fit obtained by using the previous fit to define the search area drifted too much from previous values, I would reset the search and use the convolution method (`pipeline.py` line 279).

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature for the two lane lines in lines 330 through 332 in `test_pipeline.py`. The formula for this calculation as well as the world space scaling factors were obtained from the instructional material. I used the bottom-most point of each lane line to get the radius of curvature as close to the vehicle as possible. The radius of curvature reported in the video is the average of the estimated curvature of the two lane lines. On curves, it rnaged from 1000-3000 meters, which is reasonable. On straightaways it was in the tens of kilometers and would fluctuate wildly, which is fine as the radius of curvature is highly sensitive to measurement errors in thise sections.

The calculation for the drift from lane center can be found in `test_pipeline.py` lines 168-169. This calculates the difference between the center of the warped image and the halfway points between the detected lane lines. In the final pipeline, I made sure the masking area was horizontally centered along the bottom of the image to remove any (negligible) error due to the warped image not being centered with respect to the camera view. I additionally pass in a scaling factor for transforming this image space drift to world space. The estimated drift in the video was below 0.3 meters, which again is reasonable.

The two radii of curvature and the drift can be seen in the last panel in the figure above.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 375 through 390 in my code in `test_pipeline.py`.  Here is an example of my result on a test image shown thus far:

![Figure 4][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_final.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I encountered some problems during my tests on the example images. The image below picked up extraneous pixels from the red channel at the top of the left lane line.

![Figure 4][image3]

This became more apparent when fitting the left lane line curve.

![Figure 5][image5]

And moreso when projecting the detected lane back onto the image.

![Figure 6][image7]

I tried to adjust the filtered red channel thresholds to eliminate this artifact to no avail. I also tried using only the thresholded saturation channel but that yielded worse results all around.

To avoid this problem in the future, I could threshold looking for yellow and white lane lines specifically to avoid picking up shadows and barriers as lane edges.

I also had a problem near the end of the video with shadows briefly throwing off the right lane detection. To overcome this I remembered the last six lane fitting attempts, as well as whether those fitting attempts had been accepted as within some percentage of the previous accepted fits (mentioned above). Each lane line fitting attempt was stored in a `Line` object (`pipeline.py` lines 8-60) which itself contained a history of the recent accepted `Line` objects. I found that by tuning the size of the `Line` history queue as well as the sensitivity of the acceptance threshold I could mitigate the shadow by maintaining the previousy accepted lane fits for a short period of time. This introduced a low-pass filter to the lane line fits which smoothed out good fits and rejected bad fits. If the good fit queue became depleted, a convolution search was restarted, as mentioned above.

