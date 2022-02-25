# Alcovex
Description:

  This task is related to Image Processing using Machine Learning. Create a Deep Learning model such that the modal
  should classify/identify the actual image (Real image) and photoshopped image (Fake image) when given an input
  image.
Functional Requirements:

  ● Any accuracy above 80% is acceptable.
  ● Use any predefined Deep Learning Neural Network (CNN, ANN, etc.).
Resources:

  1. Dataset - You are expected to use this dataset.
  a. The given data is raw data collected from various open/public sources (YouTube).
  b. Two random frames were extracted from each video. 41 thousand videos were processed, resulting in
  82 thousand images in the .JPG format. Each frame image is of the resolution 160 x 160.
  c. The file name can be broken down into three parts: Video’s unique string + The frame number + Label
  (can be 0 or 1). The Label is the last character in the file name. It indicates whether the frame (and the
  respective video) is modified or not.
  d. For example, if the files are named aafrqzumnqcjweerkqabfmxlyteecdou90.jpg and
  aafrqzumnqcjweerkqabfmxlyteecdou980.jpg, it means the unique string of the video is
  aafrqzumnqcjweerkqabfmxlyteecdou, 9 and 98 are the frame numbers (random numbers for each
  video) and Label is 0 (indicating not a faked video).
