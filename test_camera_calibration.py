import pickle, random, glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_cal_image_paths(folder = 'camera_cal'):
    return glob.glob(folder + '/calibration*.jpg')

with open('camera_cal.p', 'rb') as handle:
    data = pickle.load(handle)

# Load the calibration
mtx, dist = data['mtx'], data['dist']

# Choose a random image
cal_image_paths = get_cal_image_paths()
cal_image_path = random.choice(cal_image_paths)

# Load the image
cal_image = cv2.imread(cal_image_path)

# Undistort the image
undistorted = cv2.undistort(cal_image, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(cal_image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

# Save the figure
f.savefig('output_images/camera_cal.png')
plt.close(f) 
