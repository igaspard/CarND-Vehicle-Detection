## Vehicle Detection Project - Gaspard Shen

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car0.png
[image2]: ./output_images/car_not_car1.png
[image8]: ./output_images/HOG0.png
[image9]: ./output_images/HOG1.png
[image10]:./output_images/HOG2.png
[image3]: ./output_images/sliding_window.png
[image5]: ./output_images/bboxes_and_heat.png
[video1]: ./project_video_out.mp4

---
###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook.

I started by reading in all the `vehicle` and `non-vehicle` images in 1st code of the IPython notebook. Here are two examples of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `Gray` color space and tune the HOG parameters of `orientations=8`/`orientations=12`, with `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Visualize the HOG, look like when the orientations value bigger, the much obvious the vehicle's shape.

![alt text][image9]

Then tuning the `pixels_per_cell=(16, 16)`/`pixels_per_cell=(4, 4)`. Look like the using 4 can show the car much detail.
![alt text][image10]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. Since the as previous illustrate look like choose `pixels_per_cell=(4, 4)` would be good choice. However when I extract features, it take way much longer about 400 second versus pixels_per_cell is 8 only taking 165.1 second. Even worst when training, the python take over 20 GB memory can't run to the end. Then I start to record and analyze the performance and accuracy of each color features. Moreover, to exactly test on the all test images to see how different color features. And also test on the test video to see the result. Look like the YUV is one work most well at all the test images and also the videos. So I choose the YUV with original HOG parameters `orientations=8`, with `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

Here list the some logs.
###### Config 1:
179.37 Seconds to extract feature...
Using: Color Space: HSV, 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
17.16 Seconds to train SVC...
Test Accuracy of SVC =  0.9904
1.19 Seconds to Search Hot Windows...
CPU times: user 12min 33s, sys: 24.6 s, total: 12min 57s
Wall time: 8min 14s

###### Config 2:
172.4 Seconds to extract feature...
Using: Color Space: LUV 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
51.32 Seconds to train SVC...
Test Accuracy of SVC =  0.9865
1.32 Seconds to Search Hot Windows...
Video
CPU times: user 11min 56s, sys: 21.6 s, total: 12min 18s
Wall time: 7min 13s

###### Config 3:
189.88 Seconds to extract feature...
Using: Color Space: HLS, 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
22.93 Seconds to train SVC...
Test Accuracy of SVC =  0.9885
1.33 Seconds to Search Hot Windows...
Video
CPU times: user 12min, sys: 21 s, total: 12min 21s
Wall time: 7min 13s

###### Config 4:
165.1 Seconds to extract feature...
Using: Color Space: YUV, 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
47.46 Seconds to train SVC...
Test Accuracy of SVC =  0.9859
1.24 Seconds to Search Hot Windows...
CPU times: user 12min 5s, sys: 22.3 s, total: 12min 28s
Wall time: 7min 18s

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the vehicles and non-vehicle data. There are total count of 8792 vehicles and 8968 non-vehicles training sets.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the window positions at bottom part to the image which is the road and vehicles located. And then after couple try, i found the 1.5 scales look great. Ultimately I searched on one scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps and car positions:

![alt text][image5]

---
###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Sometimes I still can see the opposite car was marked even apply the heatmap and filter with the thresholded and false positives. Maybe need to dynamically choose the appropriate thresholded.

Second, some shadow of the road will be treat as the vehicles. Maybe need to collect more shadow cases non-vehicles image to improve this cases.
