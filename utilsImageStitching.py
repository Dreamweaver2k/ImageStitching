import cv2
import numpy as np
from detectBlobs import DetectBlobs
import math


# detectKeypoints(...): Detect feature keypoints in the input image
#   You can either reuse your blob detector from part 1 of this assignment
#   or you can use the provided compiled blob detector detectBlobsSolution.pyc
#
#   Input: 
#        im  - input image
#   Output: 
#        detected feature points (in any format you like).


def detectKeypoints(im):
    # YOUR CODE STARTS HERE
    blobs = DetectBlobs(im)
    return blobs


# computeDescriptors(...): compute descriptors from the detected keypoints
#   You can build the descriptors by flatting the pixels in the local 
#   neighborhood of each keypoint, or by using the SIFT feature descriptors from
#   OpenCV (see computeSIFTDescriptors(...)). Use the detected blob radii to
#   define the neighborhood scales.
#
#   Input:
#        im          - input image
#        keypoints   - detected feature points
#
#   Output:
#        descriptors - n x dim array, where n is the number of keypoints 
#                      and dim is the dimension of each descriptor. 
#
def computeDescriptors(im, keypoints):
    # YOUR CODE STARTS HERE
    return computeSIFTDescriptors(im, keypoints)


# computeSIFTDescriptors(...): compute SIFT feature descriptors from the
#   detected keypoints. This function is provided to you.
#
#   Input:
#        im          - H x W array, the input image
#        keypoints   - n x 4 array, where there are n blobs detected and
#                      each row is [x, y, radius, score]
#
#   Output:
#        descriptors - n x 128 array, where n is the number of keypoints
#                      and 128 is the dimension of each descriptor.
#
def computeSIFTDescriptors(im, keypoints):
    kp = []
    for blob in keypoints:
        kp.append(cv2.KeyPoint(blob[1], blob[0], _size=blob[2]*2, _response=blob[3]*10, _class_id=len(kp)))
    detector = cv2.xfeatures2d_SIFT.create()
    return detector.compute(im, kp)[1]


# getMatches(...): match two groups of descriptors.
#
#   There are several strategies you can use to match keypoints from the left
#   image to those in the right image. Feel free to use any (or combinations
#   of) strategies:
#
#   - Return all putative matches. You can select all pairs whose
#   descriptor distances are below a specified threshold,
#   or select the top few hundred descriptor pairs with the
#   smallest pairwise distances.
#
#   - KNN-Match. For each keypoint in the left image, you can simply return the
#   the K best pairings with keypoints in the right image.
#
#   - Lowe's Ratio Test. For each pair of keypoints to be returned, if the
#   next best match with the same left keypoint is nearly as good as the
#   current match to be returned, then this match should be thrown out.
#   For example, given point A in the left image and its best and second best
#   matches B and C in the right image, we check: score(A,B) < score(A,C)*0.75
#   If this test fails, don't return pair (A,B)
#
#
#   Input:
#         descriptors1 - the descriptors of the first image
#         descriptors2 - the descriptors of the second image
# 
#   Output: 
#         index1       - 1-D array contains the indices of descriptors1 in matches
#         index2       - 1-D array contains the indices of descriptors2 in matches
def getMatches(descriptors1, descriptors2):
    # YOUR CODE STARTS HERE
    threshold = 175
    xrange = np.shape(descriptors1)[0]
    yrange = np.shape(descriptors2)[0]
    index1 = []
    index2 = []
    values = []

    # iterate through all descriptors and find matches within a range
    for i in range(0,xrange):
        for j in range(0, yrange):
            value = np.sqrt(np.sum((descriptors1[i,:] - descriptors2[j,:]) ** 2))
            if value <= threshold and value > 0:
                index1.append(i)
                index2.append(j)
                values.append(values)

    # eliminate matches if too many
    if len(values) > 150:
        while len(values) > 150:
            remove_index = values.index(max(values))
            index1.pop(remove_index)
            index2.pop(remove_index)
            values.pop(remove_index)

    return index1, index2

# RANSAC(...): run the RANSAC algorithm to estimate a homography mapping between two images.
#   Input:
#        matches - two 1-D arrays that contain the indices on matches. 
#        keypoints1       - keypoints on the left image
#        keypoints2       - keypoints on the right image
#
#   Output:
#        H                - 3 x 3 array, a homography mapping between two images
#        numInliers       - int, the number of inliers 
#
#   Note: Use four matches to initialize the homography in each iteration.
#         You should output a single transformation that gets the most inliers 
#         in the course of all the iterations. For the various RANSAC parameters 
#         (number of iterations, inlier threshold), play around with a few 
#         "reasonable" values and pick the ones that work best.

def RANSAC(matches, keypoints1, keypoints2):
    # YOUR CODE STARTS HERE
    n = float('inf')
    p = .99
    s = 4
    sample_count = 0
    max_inliers = 0

    while n > sample_count:
        # generate random subset
        index = np.random.randint(0, np.size(matches[0]), s)
        left = []
        right = []
        for i in range(0, s):
            left.append(matches[0][index[i]])
            right.append(matches[1][index[i]])
        subset = (left, right)


        # get homography matrix of subset
        a = getA(subset, keypoints1, keypoints2)
        u,s_matrix,v = np.linalg.svd(np.matmul(a.transpose(), a))
        homography = v[-1].reshape((3,3))

        # Compute n
        inliers = get_inliers(homography, matches, keypoints1, keypoints2)
        if inliers > max_inliers:
            max_inliers = inliers
            best_homography = homography
            e = 1 - inliers/np.size(matches[0])
            if e != 1:
                n = compute_N(s, p, e)
        sample_count = sample_count + 1

    return best_homography, max_inliers



# warpImageWithMapping(...): warp one image using the homography mapping and
#   composite the warped image and another image into a panorama.
# 
#   Input: 
#        im_left, im_right - input images.
#        H                 - 3 x 3 array, a homography mapping
#  
#   Output:
#        Panorama made of the warped image and the other.
#
#       To display the full warped image, you may want to modify the matrix H.
#       CLUE: first get the coordinates of the corners under the transformation,
#             use the new corners to determine the offsets to move the
#             warped image such that it can be displayed completely.
#             Modify H to fulfill this translate operation.
#       You can use cv2.warpPerspective(...) to warp your image using H

def warpImageWithMapping(im_left, im_right, H):
    # YOUR CODE STARTS HERE
    height, width = np.shape(im_right)
    top_left = np.dot(np.linalg.inv(H), toHomogeneous((0, 0)))
    top_left = top_left[0:2] / top_left[2]
    low_left = np.dot(np.linalg.inv(H), toHomogeneous((0, height)))
    low_left = low_left[0:2] / low_left[2]
    top_right = np.dot(np.linalg.inv(H), toHomogeneous((width, 0)))
    top_right = top_right[0:2] / top_right[2]
    low_right = np.dot(np.linalg.inv(H), toHomogeneous((width, height)))
    low_right = low_right[0:2] / low_right[2]


    ytranslate = top_left[0]
    y = abs(ytranslate)
    xtranslate = top_left[1]

    translate_matrix = translation(xtranslate, ytranslate)
    H = np.linalg.inv(H)
    H = np.dot(translate_matrix, H)
    warped_imager = cv2.warpPerspective(np.transpose(im_right), H, (im_left.shape[1]+im_left.shape[0], im_left.shape[1]+im_left.shape[0]), flags= 0, borderMode=0)
    warped_imager = np.transpose(warped_imager)
    translate_matrix = translation(xtranslate, ytranslate)
    im_left_t = np.transpose(cv2.warpPerspective(np.transpose(im_left), translate_matrix, warped_imager.shape, flags=0,borderMode=0))
    new_image = im_left_t + warped_imager
    height, width = new_image.shape
    for i in range(width):
        for j in range(height):
            if new_image[j][i] != warped_imager[j][i] and im_left_t[j][i] != new_image[j][i]:
                new_image[j][i] = (int(warped_imager[j][i]) + int(im_left_t[j][i]))/2
    return new_image[:max(im_left.shape[0], im_right.shape[0]) + int(y),:]


# drawMatches(...): draw matches between the two images and display the image.
#
#   Input:
#         im1: input image on the left
#         im2: input image on the right
#         matches: (1-D array, 1-D array) that contains indices of descriptors in matches
#         keypoints1: keypoints on the left image
#         keypoints2: keypoints on the right image
#         title: title of the displayed image.
#
#   Note: This is a utility function that is provided to you. Feel free to
#   modify the code to adapt to the keypoints and matches in your own format.

def drawMatches(im1, im2, matches, keypoints1, keypoints2, title='matches'):
    idx1, idx2 = matches
    
    cv2matches = []
    for i,j in zip(idx1, idx2):
        cv2matches.append(cv2.DMatch(i, j, _distance=0))

    _kp1, _kp2 = [], []
    for i in range(keypoints1.shape[0]):
        _kp1.append(cv2.KeyPoint(keypoints1[i][1], keypoints1[i][0], _size=keypoints1[i][2], _response=keypoints1[i][3], _class_id=len(_kp1)))
    for i in range(keypoints2.shape[0]):
        _kp2.append(cv2.KeyPoint(keypoints2[i][1], keypoints2[i][0], _size=keypoints2[i][2], _response=keypoints2[i][3], _class_id=len(_kp2)))
    
    im_matches = np.empty((max(im1.shape[0], im2.shape[0]), im1.shape[1]+im2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(im1, _kp1, im2, _kp2, cv2matches, im_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Panorama', im_matches)
    cv2.waitKey()

# Find the number of points that correctly map given a homography matrix
def get_inliers(homography, matches, keypoints1, keypoints2):
    inliers = 0
    threshold = .25
    left_index = matches[0]
    right_index = matches[1]

    for i in range(len(left_index)):
        left = keypoints1[left_index[i]][0:2]
        left = toHomogeneous(left)

        right = keypoints2[right_index[i]][0:2]
        transformed = np.dot(homography, left)
        if transformed[2] != 0:
            transformed = transformed[0:2]/transformed[2]
        distance = (transformed[0] - right[0])**2 + (transformed[1] - right[1])**2

        if distance <= threshold:
            inliers = inliers + 1
    return inliers


# Compute the number of iterations necessary to be performed to have p chance of having all correct mappings
def compute_N(s, p, e):
    denom = 1-(1-e)**s
    n = math.log(1 - p)/math.log(denom)
    return math.log(1 - p)/math.log(1-(1-e)**s)


# Converts x,y coordinates to homogenous coordinates
def toHomogeneous(points):
        coordinates = np.array([points[0], points[1], 1])
        return coordinates.transpose()


# Returns a translation matrix given the offsets
def translation(x,y):
    translation = np.zeros((3, 3))
    translation[0][0] = 1
    translation[1][1] = 1
    translation[2][2] = 1
    if y < 0:
        translation[0][2] = -1 * y
    if x < 0:
        translation[1][2] = -1 * x
    return translation


# Find the A matrix to compute homography matrix
def getA(matches, keypoints1, keypoints2):
        a_matrix = np.zeros((2*len(matches[0]), 9))
        left_index = matches[0]
        right_index = matches[1]
        index = 0

        for i in range(len(left_index)):
            left = toHomogeneous(keypoints1[left_index[i]][0:2])
            right = keypoints2[right_index[i]][0:2]
            values =  -1*right[0] * np.transpose(left)
            transpose = np.transpose(left)

            a_matrix[index+1] = [transpose[0], transpose[1], transpose[2], 0, 0, 0,  values[0], values[1], values[2]]
            values = -1*right[1] * np.transpose(left)
            a_matrix[index] = [0, 0, 0,transpose[0], transpose[1], transpose[2],  values[0], values[1], values[2]]
            index = index + 2
        return a_matrix

