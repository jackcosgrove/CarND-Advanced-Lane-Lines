import pickle, glob, re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

# Construct an image mask from the lane windows detected by convolution
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
    left_fit = np.polyfit(lefty*y_per_pix, leftx*x_per_pix, 2)
    right_fit = np.polyfit(righty*y_per_pix, rightx*x_per_pix, 2)

    next_img = np.dstack((warped, warped, warped))*255
    next_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    next_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return left_fit, right_fit, next_img

def get_lane_offset(image_width, left_fitx, right_fitx, x_per_pix = 1):
    return (left_fitx[-1] + (right_fitx[-1] - left_fitx[-1])/2 - image_width/2) * x_per_pix

def generate_fitted_values(image_height, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, image_height-1, image_height)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return ploty, left_fitx, right_fitx

def get_test_image_paths(search_exp):
    return glob.glob(search_exp)

def process_test_images(test_image_paths, output_name):
    index_re = re.compile('\d+')
    font_size = 10
    
    with open('camera_cal.p', 'rb') as handle:
        data = pickle.load(handle)

    # Load the calibration
    mtx, dist = data['mtx'], data['dist']

    # Create the perspective transform and its inverse
    image_size = (600,1200)
    M, Minv = create_perspective_transforms(image_size)
    
    for test_image_path in test_image_paths:
        # Extract the test image index
        index = index_re.findall(test_image_path)[0]
        
        # Load the image
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Undistort the image
        undistorted = cv2.undistort(test_image, mtx, dist, None, mtx)
        
        color_binary, combined_binary = threshold_image(undistorted)
        
        # Plotting thresholded images
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24,9))
        f.tight_layout()
        
        ax1.imshow(test_image)
        ax1.set_title('Original Image', fontsize=font_size)

        ax2.imshow(undistorted)
        ax2.set_title('Undistorted Image', fontsize=font_size)
        
        ax3.set_title('Stacked thresholds', fontsize=font_size)
        ax3.imshow(color_binary)

        ax4.set_title('Combined S channel and gradient thresholds', fontsize=font_size)
        ax4.imshow(combined_binary, cmap='gray')

        # Save the figure
        f.savefig('output_images/' + output_name + index + '.png')
        plt.close(f)

        # Generate the warped image
        warped = cv2.warpPerspective(combined_binary, M, image_size, flags=cv2.INTER_LINEAR)

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
                    
            # Draw the results
            template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
            zero_channel = np.zeros_like(template) # create a zero color channle 
            template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
            detected = cv2.addWeighted(detected, 1, template, 0.5, 0.0) # overlay the orignal

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

        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Now test fitting the same image again using the curve as a window spine
        margin = 100
        next_left_fit, next_right_fit, next_img = find_next_lane_line(warped, left_fit, right_fit, margin)
        mask = np.zeros_like(next_img)
        
        next_ploty, next_left_fitx, next_right_fitx = generate_fitted_values(warped.shape[0], left_fit, right_fit)
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([next_left_fitx-margin, next_ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([next_left_fitx+margin, next_ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([next_right_fitx-margin, next_ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([next_right_fitx+margin, next_ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(mask, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(mask, np.int_([right_line_pts]), (0,255, 0))
        mask = cv2.addWeighted(next_img, 1, mask, 0.3, 0)

        # Calculate the radius of curvature with respect to the bottom of the image
        # which is the car dashboard
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        left_fit_cr, right_fit_cr, curvature = find_next_lane_line(warped, next_left_fit, next_right_fit, margin, ym_per_pix, xm_per_pix)
        
        # Calculate the new radii of curvature
        y_eval = np.max(next_ploty)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Calculate the lane offset
        lane_offset = get_lane_offset(warped.shape[1], next_left_fitx, next_right_fitx, xm_per_pix)
                        
        fitted = np.zeros_like(detected)
        # Color the detected left lane markings red
        fitted[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # Color the detected right lane markings blue
        fitted[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        

        # Plot the warped image, detected lane windows, and fitted lane lines with measurements
        f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24,9))
        f.tight_layout()

        ax1.set_title('Warped Image', fontsize=font_size)
        ax1.imshow(warped, cmap='gray')
        
        ax2.set_title('Detected Lane Windows', fontsize=font_size)
        ax2.imshow(detected)

        ax3.set_title('Fitted Lane Lines', fontsize=font_size)
        ax3.imshow(fitted)
        ax3.plot(left_fitx, ploty, color='yellow')
        ax3.plot(right_fitx, ploty, color='yellow')

        ax4.set_title('Detected Lane Lines from Fit', fontsize=font_size)
        ax4.imshow(mask)
        ax4.plot(next_left_fitx, next_ploty, color='yellow')
        ax4.plot(next_right_fitx, next_ploty, color='yellow')

        ax5.set_title('Measurements', fontsize=font_size)
        ax5.imshow(curvature)
        ax5.plot(next_left_fitx, next_ploty, color='yellow')
        ax5.plot(next_right_fitx, next_ploty, color='yellow')
        ax5.text(int(curvature.shape[1]/4), int(curvature.shape[0]/4), 'Left: ' + str(left_curverad) + ' m', fontsize=font_size, color='yellow')
        ax5.text(int(curvature.shape[1]/4), int(curvature.shape[0]*3/4), 'Right: ' + str(right_curverad) + ' m', fontsize=font_size, color='yellow')
        ax5.text(int(curvature.shape[1]/4), int(curvature.shape[0]/2), 'Offset: ' + str(lane_offset) + ' m', fontsize=font_size, color='yellow')
        
        f.savefig('output_images/' + output_name + '_warped' + index + '.png')
        plt.close(f)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([next_left_fitx, next_ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([next_right_fitx, next_ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (test_image.shape[1], test_image.shape[0])) 
        # Combine the result with the original image
        final = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

        # Plot the final image
        f, (ax1) = plt.subplots(1, 1, figsize=(24,9))
        f.tight_layout()

        ax1.set_title('Final Image', fontsize=font_size)
        ax1.imshow(final)
        ax1.text(int(curvature.shape[1]/4), int(curvature.shape[0]/50), 'RoC: ' + str((left_curverad + right_curverad)/2.) + ' m', fontsize=font_size*2, color='yellow')
        ax1.text(int(curvature.shape[1]/4), int(curvature.shape[0]/25), 'Offset: ' + str(lane_offset) + ' m', fontsize=font_size*2, color='yellow')

        f.savefig('output_images/' + output_name + '_final' + index + '.png')
        plt.close(f)
        

test_image_paths = get_test_image_paths('test_images/test*.jpg')
process_test_images(test_image_paths, 'test')

test_image_paths = get_test_image_paths('test_images/straight_lines*.jpg')
process_test_images(test_image_paths, 'straight_lines')
