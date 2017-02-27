import pickle, glob, re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

class Line():
    def __init__(self, image_height):
        self.ploty = np.linspace(0, image_height-1, image_height)
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = None
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        # fit coefficients of the last n fits of the line
        self.recent_fit = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

    def inherit_fit(self, previous, recent_line_count):
        if previous is None:
            return
        
        if previous.recent_xfitted is not None:
            if previous.recent_xfitted.shape[1] >= recent_line_count:
                self.recent_xfitted = previous.recent_xfitted[:,1:]
            else:
                self.recent_xfitted = previous.recent_xfitted
            self.bestx = np.average(self.recent_xfitted, axis=1)

        if previous.recent_fit is not None:
            if previous.recent_fit.shape[1] > recent_line_count:
                self.recent_fit = previous.recent_fit[:,1:]
            else:
                self.recent_fit = previous.recent_fit
            self.best_fit = np.average(self.recent_fit, axis=1)
        
    def append_fit(self, fit, recent_line_count, diff_threshold = 2):
        # Generate x and y values for plotting
        xfitted = fit[0]*self.ploty**2 + fit[1]*self.ploty + fit[2]
        
        if self.best_fit is not None:
            percent_diff = np.average(np.abs((self.best_fit - fit) / self.best_fit))
            
            if percent_diff < diff_threshold or self.recent_fit.shape[1] < recent_line_count:
                self.detected = True
                self.recent_fit = np.concatenate((self.recent_fit, np.expand_dims(fit, axis=1)), axis=1)
                self.best_fit = np.average(self.recent_fit, axis=1)
                self.recent_xfitted = np.concatenate((self.recent_xfitted, np.expand_dims(xfitted, axis=1)), axis=1)
                self.bestx = np.average(self.recent_xfitted, axis=1)
            else:
                print("Missed frame")
        else:
            self.detected = True
            self.recent_fit = np.expand_dims(fit, axis=1)
            self.best_fit = fit
            self.recent_xfitted = np.expand_dims(xfitted, axis=1)
            self.bestx = xfitted

global recent_lines
recent_lines = []
        
def threshold_image(image, r_thresh_min = 40, r_thresh_max = 100, s_thresh_min = 170, s_thresh_max = 255):
    # Separate the red channel
    r_channel = image[:,:,0]
    
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(r_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.abs(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= r_thresh_min) & (scaled_sobel <= r_thresh_max)] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Stack each channel to view their individual contributions in red and green respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((255*sxbinary, 255*s_binary, np.zeros_like(sxbinary)))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return color_binary, combined_binary

def create_perspective_transforms(image_size):

    src = np.float32([[575, 450],
                      [725, 450],
                      [1250, 700],
                      [30, 700]]) # Keep the same horizontal margin (30) at the bottom for lane offset calculation
    dst = np.float32([[0, 0],
                      [image_size[0], 0],
                      [image_size[0], image_size[1]],
                      [0, image_size[1]]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    bottom, top, left, right = (int(img_ref.shape[0]-(level+1)*height), int(img_ref.shape[0]-level*height), max(0,int(center-width/2)), min(int(center+width/2),img_ref.shape[1]))
    output[bottom:top,left:right] = 1
    return output, bottom, top, left, right

def find_left_window_centroid(l_sum, window, window_width, threshold):
    l_conv = np.convolve(window,l_sum)
    l_max = np.argmax(l_conv)
    if l_conv[l_max] < threshold:
        l_center = None
    else:
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_center = l_max-window_width/2 + offset
        
    return l_center

def find_right_window_centroid(r_sum, window, window_width, threshold, image_width):
    r_conv = np.convolve(window,r_sum)
    r_max = np.argmax(r_conv)

    if r_conv[r_max] < threshold:
        r_center = None
    else:
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        r_center = r_max-window_width/2+int(image_width/2) + offset

    return r_center

# Find the centroids of a set of windows that overlay probable lane marking areas
def find_window_centroids(warped, window_width, window_height, margin, threshold_pct):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 

    threshold = window_width * threshold_pct * 255
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    
    l_center = find_left_window_centroid(l_sum, window, window_width, threshold)
    r_center = find_right_window_centroid(r_sum, window, window_width, threshold, warped.shape[1])
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        section = warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:]
        image_layer = np.sum(section, axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        
        if l_center is not None:
            l_min_index = int(max(l_center-margin,0))
            l_max_index = int(min(l_center+margin,warped.shape[1]))
            l_max = np.argmax(conv_signal[l_min_index:l_max_index])
            if conv_signal[l_max] < threshold:
                l_center = None
            else:
                l_center = l_max+l_min_index
        else:
            l_sum = np.sum(section[:,:int(section.shape[1]/2)], axis=0)
            l_center = find_left_window_centroid(l_sum, window, window_width, threshold)
                
        # Find the best right centroid by using past right center as a reference
        if r_center is not None:
            r_min_index = int(max(r_center-margin,0))
            r_max_index = int(min(r_center+margin,warped.shape[1]))
            r_max = np.argmax(conv_signal[r_min_index:r_max_index])
            if conv_signal[r_max] < threshold:
                r_center = None
            else:
                r_center = r_max+r_min_index
        else:
            r_sum = np.sum(section[:,int(section.shape[1]/2):], axis=0)
            r_center = find_right_window_centroid(r_sum, window, window_width, threshold, warped.shape[1])
                
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

def find_next_lane_line(warped, left_fit, right_fit, margin, y_per_pix = 1, x_per_pix = 1):
    if left_fit is None or right_fit is None:
        return None, None
    
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
        return None, None
           
    left_fit = np.polyfit(lefty*y_per_pix, leftx*x_per_pix, 2)
    right_fit = np.polyfit(righty*y_per_pix, rightx*x_per_pix, 2)

    return left_fit, right_fit

def get_lane_offset(image_width, left_fitx, right_fitx, x_per_pix = 1):
    return (left_fitx[-1] + (right_fitx[-1] - left_fitx[-1])/2 - image_width/2) * x_per_pix

def process_image(image):
    with open('camera_cal.p', 'rb') as handle:
        data = pickle.load(handle)

    # Load the calibration
    mtx, dist = data['mtx'], data['dist']

    # Create the perspective transform and its inverse
    image_size = (600,1200)
    M, Minv = create_perspective_transforms(image_size)

    # Undistort the image
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    
    color_binary, combined_binary = threshold_image(undistorted)

    # Generate the warped image
    warped = cv2.warpPerspective(combined_binary, M, image_size, flags=cv2.INTER_LINEAR)

    margin = 100
    recent_line_count = 8
    
    global recent_lines

    detected_lines = [line_pair for line_pair in recent_lines if (line_pair[0].detected and line_pair[1].detected)]
   
    if len(detected_lines) > 0:
        last_left_line, last_right_line = detected_lines[-1]
    else:
        last_left_line = None
        last_right_line = None
    
    left_line = Line(warped.shape[0])
    left_line.inherit_fit(last_left_line, recent_line_count)
    
    right_line = Line(warped.shape[0])
    right_line.inherit_fit(last_right_line, recent_line_count)
    
    # If we've already fitted lane line curves, use them to find the next curves
    if last_left_line is not None and last_right_line is not None:
        left_fit, right_fit = find_next_lane_line(warped, last_left_line.best_fit, last_right_line.best_fit, margin)

        if left_fit is not None and right_fit is not None:
            left_line.append_fit(left_fit, recent_line_count)
            right_line.append_fit(right_fit, recent_line_count)
        
    # else use convolution to detect the lane lines initially
    else:
        print("Resetting lane lines...")
        # How wide the search window will be
        window_width = 100
        # How tall the search window will be - 720 pixel height produces 9 vertical layers
        window_height = 80
        # How much to slide left and right for searching
        margin = 100
        # Fraction of window width to use as a threshold for detecting a lane line
        threshold_pct = 0.01
        
        # Now detect lane lines using convolution
        window_centroids = find_window_centroids(warped, window_height, window_height, margin, threshold_pct)

        detected = 255*np.array(cv2.merge((warped,warped,warped)),np.uint8) # making the original road pixels 3 color channels

        # Arrays to receive the lane marking indexes
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        # If we found any window centers
        if len(window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            r_points = np.zeros_like(warped)

            # Go through each level and draw the windows    
            for level in range(0,len(window_centroids)):
                if window_centroids[level][0] is not None:
                    l_mask, win_y_low, win_y_high, win_x_low, win_x_high = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
                    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                    left_lane_inds.append(good_left_inds)
                    
                if window_centroids[level][1] is not None:
                    r_mask, win_y_low, win_y_high, win_x_low, win_x_high = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
                    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255
                    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                    right_lane_inds.append(good_right_inds)

        if len(left_lane_inds) > 0 and len(right_lane_inds) > 0:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)

            # Extract left and right line pixel positions
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds] 

            # Fit a second order polynomial to each
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

            left_line.append_fit(left_fit, recent_line_count)
            right_line.append_fit(right_fit, recent_line_count)

    recent_lines.append((left_line, right_line))
    if len(recent_lines) > recent_line_count:
        recent_lines.pop(0)

    # Calculate the radius of curvature with respect to the bottom of the image
    # which is the car dashboard
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    left_fit_cr, right_fit_cr = find_next_lane_line(warped, left_line.best_fit, right_line.best_fit, margin, ym_per_pix, xm_per_pix)

    if left_fit_cr is not None and right_fit_cr is not None:
        # Calculate the new radii of curvature
        y_eval = np.max(left_line.ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        roc = str((left_curverad + right_curverad)/2.)
    else:
        roc = '---'

    if left_line.bestx is not None and right_line.bestx is not None:
        # Calculate the lane offset
        lane_offset = str(get_lane_offset(warped.shape[1], left_line.bestx, right_line.bestx, xm_per_pix))
    else:
        lane_offset = '---'

    if left_line.bestx is not None and right_line.bestx is not None:
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()

        pts_left = np.array([np.transpose(np.vstack([left_line.bestx, left_line.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, right_line.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        final = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)
    else:
        final = undistorted

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final, 'RoC: ' + roc + ' m', (50, 25), font, 1, (255,255,255))
    cv2.putText(final,  'Offset: ' + lane_offset + ' m', (50, 50), font, 1, (255,255,255))
    
    return final

output = 'project_video_final.mp4'
clip1 = VideoFileClip("project_video.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)


