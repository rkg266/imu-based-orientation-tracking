# imu-based-orientation-tracking
Implemented a projected gradient descent algorithm to track the 3D-orientation of a robot undergoing pure rotation using the readings from the IMU.

## Tasks:
1. Estimate the orientation trajectory of the robot by minimizing the squared error between Kinematics model and the IMU measurements. Compare the estimated trajectory with ground truth obtained from VICON recordings. <br>
**Results:** The estimated and ground truth values of the Roll-Pitch-Yaw are plotted across time for nine different datasets.
<br> Vicon: Ground truth
![Roll-Pitch-Yaw plots](/plots_images/RollPitchYaw_plots.png)

2. Given images of surroundings captured by an onboard camera. With the estimated orientation trajectory, stitch all the images to create a panoramic view of the robot throughout the duration of the experiment  <br>
**Results:** Panoramas generated for the datasets #8 and #9.
![Panaroma of dataset #8](/plots_images/Train_data_8_pan.jpg) <br>
![Panaroma of dataset #9](/plots_images/Train_data_9_pan.jpg)

## Data:
Contact rkgutta266@gmail.com

## Libraries required:
Numpy, matplotlib, pickle, os, autograd, transforms3d, tqdm

## Running code:
1. Open main_code.py
2. Look for ##### USER EDITABLE ##### section in the top.
3. Please update the MANDATORY file paths and flags.
4. Various flags are provided to switch ON/OFF - parameters estimation, graphs plotting and panorama.
5. Parameters estimation generates pickle files of the estimated parameters, only after which you can perform graphs plotting and panorama.
4. Run main_code.py

## Information about other folders:
1. //trainpickle: Contains pickle files of estimated quaternions, cost values and gradient norm values of training data. (FILE INDEXING: 0...8)
2. //testpickle: Contains pickle files of estimated quaternions, cost values and gradient norm values of testing data. (FILE INDEXING: 9, 10)
3. //PlotImages: Contains Yaw-Pitch-Roll and cost function plots for all data sets. (FILE INDEXING: 1...11)
4. //PanoramaImages: Contains panorama images for all data sets. (FILE INDEXING: 1...11)


