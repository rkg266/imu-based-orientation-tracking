# imu-based-orientation-tracking
Implemented a projected gradient descent algorithm to track the 3D-orientation of a robot using the readings from the IMU

Caution:
1. Please DO NOT merge training and test data. Store them in completely different directories. Also DO NOT change the .pkl file names of the data.

Libraries required:
Numpy, matplotlib, pickle, os, autograd, transforms3d, tqdm

Running code:
1. Open ece276_code5.py
2. Look for ##### USER EDITABLE ##### section in the top.
3. Please update the MANDATORY file paths and flags.
4. Various flags are provided to switch ON/OFF - parameters estimation, graphs plotting and panorama.
5. Parameters estimation generates pickle files of the etimated parameters, only after which you can perform graphs plotting and panorama.
4. Run ece276_code5.py
