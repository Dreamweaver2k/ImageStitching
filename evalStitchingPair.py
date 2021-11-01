import os, sys
import cv2
import numpy as np
from utilsImageStitching import *
import matplotlib.pyplot as plt

# Load the left image and the right image.
imagePathLeft = 'C:\\Users\\Chand\\Documents\\Princeton\\FALL2020\\COS_429\\projects\\COS429HW1\\data\\image_sets\\pier\\1.JPG'
imagePathRight = 'C:\\Users\\Chand\\Documents\\Princeton\\FALL2020\\COS_429\\projects\\COS429HW1\\data\\image_sets\\pier\\2.JPG'
imagePathMid = 'C:\\Users\\Chand\\Documents\\Princeton\\FALL2020\\COS_429\\projects\\COS429HW1\\data\\image_sets\\pier\\3.JPG'

im_left = cv2.imread(imagePathLeft, cv2.IMREAD_GRAYSCALE)
im_right = cv2.imread(imagePathRight, cv2.IMREAD_GRAYSCALE)
im_mid = cv2.imread(imagePathMid, cv2.IMREAD_GRAYSCALE)

# Implement the detectKeypoints() function in utilsImageStitching.py
# to detect feature points for both images.
print('keypoints')
keypoints_left = detectKeypoints(im_left/255)
keypoints_right = detectKeypoints(im_right/255)
keypoints_mid = detectKeypoints(im_mid/255)

# Implement the computeDescriptors() function in utilsImageStitching.py
# to compute descriptors on keypoints
print('descriptors')
descriptors_left = computeDescriptors(im_left, keypoints_left)
descriptors_right = computeDescriptors(im_right, keypoints_right)
descriptors_mid = computeDescriptors(im_mid, keypoints_mid)

# Implement the getMatches() function in utilsImageStitching.py
# to get matches
print('matches....')
matches = getMatches(descriptors_left, descriptors_right)


drawMatches(im_left, im_right, matches, keypoints_left, keypoints_right)

# Implement the RANSAC() function in utilsImageStitching.py.
# Run RANSAC to estimate a homography mapping
print('ransac....')
H, numInliers = RANSAC(matches, keypoints_left, keypoints_right)
print(H)

# Implement warpImageWithMapping() function in utilsImageStitching.py.
# Warp one image with the estimated homography mapping
# and composite the warpped image and another one.
panorama = warpImageWithMapping(im_left, im_right, H)
plt.imshow(panorama, cmap='gray', vmin=0, vmax=255)

plt.show()



key_pan = detectKeypoints(panorama/255)
desc_pan = computeDescriptors(panorama, key_pan)
matches_pan = getMatches(descriptors_mid,desc_pan)
drawMatches(im_mid, panorama, matches_pan, keypoints_mid, key_pan)
print('number of matches: %d' %np.size(matches_pan[0]))
#drawMatches(im_mid, panorama, matches_pan,keypoints_mid,key_pan)
H, inliers = RANSAC(matches_pan, keypoints_mid, key_pan)
panorama =warpImageWithMapping(im_mid, panorama, H)

plt.imshow(panorama, cmap='gray', vmin=0, vmax=255)

plt.show()