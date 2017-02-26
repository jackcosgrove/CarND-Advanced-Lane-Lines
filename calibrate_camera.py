import pickle, glob
import numpy as np
import cv2

def get_cal_image_paths(folder = 'camera_cal'):
    return glob.glob(folder + '/calibration*.jpg')

def calibrate_camera(cal_image_paths, nx = 9, ny = 6):
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane

    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates

    for cal_image_path in cal_image_paths:
        # Read the image
        cal_image = cv2.imread(cal_image_path)

        # Convert image to gray scale
        gray = cv2.cvtColor(cal_image, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist

# Make a list of calibration images
cal_image_paths = get_cal_image_paths()

# Calibrate the camera
mtx, dist = calibrate_camera(cal_image_paths)

data = { 'mtx': mtx, 'dist': dist }

with open('camera_cal.p', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
