Speed Challenge
======

Predicts the speed of a vehicle from dashcam footage
Written in python 3 with Pytorch and OpenCV

image quality is downgraded by 50% and then parsed into OpenCV's Farneback Dense Optical Flow 

The input to the NN is the black and white optical flow image

MSE on the train video appears to be about ~7, although it could be overfitting and/or I calculated MSE wrong

improvements to be made are 

- crop the image to remove background noise
- normalise brightness of each frame (bright sky might be noise, other people have done this)
- try training the net with full_size images (also maybe the colour from the optical flow thing is useful)

TODO: 
- make improvements
- optimise data processing
- refactor and remove repeated code


another approach
------

I read or heard something, somewhere about something called a "Depth Net", for depth estimation, which might be a better 
approach for this problem 

readme from forked repo with data below |
                                        V

Welcome to the comma.ai Programming Challenge!
======

Your goal is to predict the speed of a car from a video.

- data/train.mp4 is a video of driving containing 20400 frames. Video is shot at 20 fps.
- data/train.txt contains the speed of the car at each frame, one speed on each line.
- data/test.mp4 is a different driving video containing 10798 frames. Video is shot at 20 fps.

Deliverable
-----

Your deliverable is test.txt. E-mail it to givemeajob@comma.ai, or if you think you did particularly well, e-mail it to George.

Evaluation
-----

We will evaluate your test.txt using mean squared error. <10 is good. <5 is better. <3 is heart.
