import numpy as np
import math
import cv2
from scipy import ndimage
from matplotlib import pyplot as plt


# This code is taken and converted to Python from:
#
#   CMPSCI 670: Computer Vision, Fall 2014
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
# Part1:
#
#   DetectBlobs(...) detects blobs in the image using the Laplacian
#   of Gaussian filter. Blobs of different size are detected by scaling sigma
#   as well as the size of the filter or the size of the image. Downsampling
#   the image will be faster than upsampling the filter, but the decision of
#   how to implement this function is up to you.
#
#   For each filter scale or image scale and sigma, you will need to keep track of
#   the location and matching score for every blob detection. To combine the 2D maps
#   of blob detections for each scale and for each sigma into a single 2D map of
#   blob detections with varying radii and matching scores, you will need to use
#   Non-Max Suppression (NMS).
#
#   Additional Notes:
#       - We greyscale the input image for simplicity
#       - For a simple implementation of Non-Max-Suppression, you can suppress
#           all but the most likely detection within a sliding window over the
#           2D maps of blob detections (ndimage.maximum_filter may help).
#           To combine blob detections into a single 2D output,
#           you can take the max along the sigma and scale axes. If there are
#           still too many blobs detected, you can do a final NMS. Remember to
#           keep track of the blob radii.
#       - A tip that may improve your LoG filter: Normalize your LoG filter
#           values so that your blobs detections aren't biased towards larger
#           filters sizes
#
#   You can qualitatively evaluate your code using the evalBlobs.py script.
#
# Input:
#   im             - input image
#   sigma          - base sigma of the LoG filter
#   num_intervals  - number of sigma values for each filter size
#   threshold      - threshold for blob detection
#
# Ouput:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def DetectBlobs(
        im,
        sigma=11,
        num_octaves=1,
        num_intervals=12,
        threshold=1e-4
):
    
    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255

    # YOUR CODE STARTS HERE
    full_im = np.zeros((np.size(im[0]), np.size(im[0])))
    intermediate_im = np.zeros((np.size(im[0]), np.size(im[0])))

    ref_radius = np.zeros((np.size(im[0]), np.size(im[0])))
    radii = np.zeros((num_intervals + 1) * num_octaves)

    # Compute filters to be applied later
    size = sigma * 3

    laplacian_filter = laplacian(sigma, size)

    # apply filters for all intervals and octaves
    for i in range(0, num_octaves):
        resize_factor = 1 / (pow(2, i))
        width = int(resize_factor * np.size(im[0]))
        height = width
        dsize = (width, height)
        temp_im = cv2.resize(im, dsize)
        print(np.size(temp_im[0]))
 
        # convert back to regular size
    filter = ndimage.maximum_filter(intermediate_im, size=radii[1] * pow(2, i))
    filter = (intermediate_im - filter)
    for x in range(0, np.size(intermediate_im[0])):
        for y in range(0, np.size(intermediate_im[0])):
            if filter[x][y] < 0:
                intermediate_im[x][y] = 0
    for x in range(0, np.size(intermediate_im[0])):
        for y in range(0, np.size(intermediate_im[0])):
            if intermediate_im[x][y] >= full_im[x][y]:
                full_im[x][y] = intermediate_im[x][y]
                ref_radius[x][y] = radii[1] * pow(2, i)


filter = ndimage.maximum_filter(full_im, size=radii[i] * 3)
filter = full_im - filter
for x in range(0, np.size(full_im[0])):
    for y in range(0, np.size(full_im[0])):
        if filter[x][y] < 0:
            full_im[x][y] = 0

blob = np.where(full_im > threshold)
# print(full_im)
print(np.size(blob[0]))
blobs = np.array([[im.shape[0] * 0.5, im.shape[1] * 0.5, 0.00 * min(im.shape[0], im.shape[1]), 1]])
for x in range(0, np.size(blob[0])):
    new_blob = np.array([blob[0][x], blob[1][x], ref_radius[blob[0][x]][blob[1][x]], full_im[blob[0][x]][blob[1][x]]])
    blobs = np.vstack((blobs, new_blob))

return blobs.round()


def show(img):
    img2 = img / np.max(img)
    plt.imshow(img2, cmap='gray', clim=(0, 1), vmin=0, vmax=1)
    plt.show()


def laplacian(sigma, size):
    laplacian_filter = np.zeros((size, size))
    center = int(size / 2)
    for x in range(0, size):
        for y in range(0, size):
            laplacian_filter[x][y] = (pow(x - center, 2) + pow(y - center, 2) - 2 * pow(sigma, 2)) * math.exp(
                -1 * (pow(x - center, 2) + pow(y - center, 2)) / (2 * pow(sigma, 2)))
    return laplacian_filter


def gaussian(sigma, size):
    gaussian_filter = np.zeros((size, size))
    center = int(size / 2)
    for x in range(0, size):
        for y in range(0, size):
            gaussian_filter[x][y] = 1 / (2 * math.pi * pow(sigma, 2)) * math.exp(
                -1 * (pow(x - center, 2) + pow(y - center, 2)) / (2 * pow(sigma, 2)))
    return gaussian_filter

# YOUR CODE STARTS HERE
height, width = np.shape(im_right)[0:2]
height = height * 2
width = width * 2
dsize = (width, height)

warped_imager = cv2.warpPerspective(im_right, np.linalg.inv(H), dsize)

show(im_left)
show(warped_imager)

new_image = np.empty((max(im_left.shape[0], warped_imager.shape[0]), im_left.shape[1] + warped_imager.shape[1]),
                     dtype=np.uint8)

top_left = np.dot(np.linalg.inv(H), toHomogeneous((0, 0)))
top_left = top_left[0:2] / top_left[2]
low_left = np.dot(np.linalg.inv(H), toHomogeneous((0, height)))
low_left = low_left[0:2] / low_left[2]
top_right = np.dot(np.linalg.inv(H), toHomogeneous((width, 0)))
top_right = top_right[0:2] / top_right[2]
low_right = np.dot(np.linalg.inv(H), toHomogeneous((width, height)))
low_right = low_right[0:2] / low_right[2]
print(top_left)
print(low_left)
print(top_right)
print(low_right)
height, width = np.shape(warped_imager)[0:2]
dsize = (width, height)
translate_mat = translation()

warped_imager = cv2.warpPerspective(warped_imager, translate_mat, dsize)

new_image[:im_left.shape[0], :im_left.shape[1]] = im_left
new_image[:warped_imager.shape[0], im_left.shape[1]:] = warped_imager
height, width = warped_imager.shape

height, width = np.shape(im_right)[0:2]
height = height * 2
width = width * 2
dsize = (width, height)

top_left = np.dot(np.linalg.inv(H), toHomogeneous((0, 0)))
top_left = top_left[0:2] / top_left[2]
low_left = np.dot(np.linalg.inv(H), toHomogeneous((0, height)))
low_left = low_left[0:2] / low_left[2]
top_right = np.dot(np.linalg.inv(H), toHomogeneous((width, 0)))
top_right = top_right[0:2] / top_right[2]
low_right = np.dot(np.linalg.inv(H), toHomogeneous((width, height)))
low_right = low_right[0:2] / low_right[2]

xtranslate = min(low_right[0], low_left[0], top_right[0], top_left[0])
ytranslate = min(low_right[1], low_left[1], top_right[1], top_left[1])

print(top_left)
print(low_left)
print(top_right)
print(low_right)

height, width = np.shape(im_left)[0:2]
dsize = (width, height)
translate_mat = translation(xtranslate, ytranslate)
print(translate_mat)
H = np.dot(np.linalg.inv(H), translate_mat)

warped_imager = cv2.warpPerspective(im_right, H, (im_left.shape[1] + im_right.shape[1], im_right.shape[0]))

new_image = np.empty((warped_imager.shape[1] + im_left.shape[1], warped_imager.shape[0]))

new_image[:im_right.shape[0], :im_left.shape[1]] = im_left

plt.imshow(new_image)
plt.show()
height, width = warped_imager.shape

def warpImageWithMapping(im_left, im_right, H):
    print(H)
    H = np.linalg.inv(H)
    print(H)
    #H[[0,1]] = H[[1,0]]
    print(H)
    warped_imager = cv2.warpPerspective(im_right, H, (im_right.shape[1] + im_left.shape[1], im_right.shape[0]))
    show(warped_imager)
    # YOUR CODE STARTS HERE

    height, width = np.shape(im_right)[0:2]
    top_left = np.dot(np.linalg.inv(H), toHomogeneous((0, 0)))
    top_left = top_left[0:2] / top_left[2]
    low_left = np.dot(np.linalg.inv(H), toHomogeneous((0, height)))
    low_left = low_left[0:2] / low_left[2]
    top_right = np.dot(np.linalg.inv(H), toHomogeneous((width, 0)))
    top_right = top_right[0:2] / top_right[2]
    low_right = np.dot(np.linalg.inv(H), toHomogeneous((width, height)))
    low_right = low_right[0:2] / low_right[2]

    xtranslate = min(low_right[0], low_left[0], top_right[0], top_left[0])
    ytranslate = min(low_right[1], low_left[1], top_right[1], top_left[1])

    x = max(top_right[0], low_right[0])
    y = max(top_left[1], top_right[1])
    x = x - im_left.shape[1]
    print((1,y))
    #xtranslate = im_left.shape[1] + xtranslate
    #ytranslate = im_left.shape[0] - ytranslate
    """tmat = translation(0,im_left.shape[1] + xtranslate)

    warped_imager = cv2.warpPerspective(im_right, np.dot(np.linalg.inv(H), tmat), (im_right.shape[1] + im_left.shape[1], im_right.shape[0]))
    show(warped_imager)"""

    #xtranslate = im_left.shape[1]
    tmat = translation(0, ytranslate)
    print(tmat)
    warped_imager = cv2.warpPerspective(im_right, H, (im_right.shape[1] + im_left.shape[1], im_right.shape[0]))
    """tmat = translation(xtranslate, 0)
    im_left = cv2.warpPerspective(im_left, tmat, (im_left.shape[1], im_left.shape[0]))"""

    #warped_imager = cv2.warpPerspective(im_right, np.linalg.inv(H), (im_right.shape[1] + im_left.shape[1], im_right.shape[0] + im_left.shape[0]))

    show(warped_imager)
    #warped_imager = cv2.warpPerspective(warped_imager, tmat, (warped_imager.shape[1], warped_imager.shape[0]))

    warped_imager[:warped_imager.shape[0], :im_left.shape[1]] = im_left

    show(warped_imager)

    height, width = np.shape(im_left)[0:2]
    height = height*2
    width = width*2
    dsize = (width, height)




    top_left = np.dot(H, toHomogeneous((0, 0)))
    top_left = top_left[0:2] / top_left[2]
    low_left = np.dot(H, toHomogeneous((0, height)))
    low_left = low_left[0:2] / low_left[2]
    top_right = np.dot(H, toHomogeneous((width, 0)))
    top_right = top_right[0:2] / top_right[2]
    low_right = np.dot(H, toHomogeneous((width, height)))
    low_right = low_right[0:2] / low_right[2]

    xtranslate = max(low_right[0], low_left[0], top_right[0], top_left[0])
    ytranslate = min(low_right[1], low_left[1], top_right[1], top_left[1])
    xtranslate = im_left.shape[1]
    print(top_left)
    print(low_left)
    print(top_right)
    print(low_right)

    height, width = np.shape(im_left)[0:2]
    dsize = (width, height)
    translate_mat = translation(xtranslate, ytranslate)
    print(translate_mat)
    H = np.dot(np.linalg.inv(H), translate_mat)

    warped_imagel = cv2.warpPerspective(im_left, H,  (im_left.shape[1] + im_right.shape[1], im_right.shape[0]))
    #warped_imager = cv2.warpPerspective(im_right, translate_mat,  (im_right.shape[1], im_right.shape[0]))
    #warped_imager = cv2.warpPerspective(warped_imager, translate_mat, (warped_imager.shape[1], warped_imager.shape[0]))
    show(im_left)
    show(warped_imagel)

    new_image = warped_imagel
   # new_image[:warped_imagel.shape[0],:warped_imagel.shape[1]-im_left.shape[1]] = warped_imagel[:warped_imagel.shape[0], :warped_imagel.shape[1]-im_left.shape[1]]
    new_image[:im_right.shape[0], int(low_left[1]) + warped_imagel.shape[1]: int(low_left[1]) + warped_imagel.shape[1] + im_right.shape[1]] = im_right
    show(new_image)
    #height, width = np.shape(warped_image)


    print(H)
    #warped_imager = cv2.warpPerspective(warped_imager, translate_mat, (im_left.shape[1] + im_right.shape[1], im_left.shape[0]))
    #new_image = np.empty((warped_imagel.shape[0], warped_imagel.shape[1]))

    #new_image[:warped_imager.shape[0], (-1) * int(top_left[1]) + im_left.shape[1]:-1 * int(top_left[1])] = warped_imager[,:im_left.shape[1]]
    #new_image[:warped_imager.shape[0], im_left.shape[1] - int(top_left[1]): new_image.shape[1]-int(top_left[1])] = warped_imager
    #new_image[:warped_imager.shape[0], 222:] = warped_imager
    new_image[:warped_imagel.shape[0], :warped_imagel.shape[1]] = warped_imagel
    new_image[:warped_imagel.shape[0], warped_imagel.shape[1] - im_right.shape[1]-430:-430] = im_right[:,430:]

    # new_image[int(top_left[0]):int(top_left[0]) + warped_imager.shape[0], im_left.shape[1] - int(top_left[1]):int(-1*top_left[1])] = warped_imager
    # new_image[int(top_left[0]):int(top_left[0])+warped_imager.shape[0], int(top_right[1]):warped_imager.shape[1]+int(1*top_right[1])] = warped_imager
    #new_image[:warped_imager.shape[0], im_left.shape[1]:] = warped_imager
    #new_image[:warped_imager.shape[0], int(top_right[1]) + 20:warped_imager.shape[1] + int(top_right[1]) + 20] = warped_imager
    #new_image[36:36+warped_imager.shape[0], int(top_right[1]) + 20:warped_imager.shape[1] + int(top_right[1]) + 20] = warped_imager

    plt.imshow(new_image)
    plt.show()
    height, width = warped_imagel.shape
    print(warped_imagel.shape[1])

    return new_image




 a_matrix = np.zeros((2*len(matches[0]), 9))
        left_index = matches[0]
        right_index = matches[1]
        index = 0

        for i in range(len(left_index)):
            left = toHomogeneous(keypoints1[left_index[i]][0:2])
            right = keypoints2[right_index[i]][0:2]
            values =  right[1] * np.transpose(left)
            transpose = -1*np.transpose(left)

            a_matrix[index] = [0 , 0, 0, transpose[0], transpose[1], transpose[2], values[0], values[1], values[2]]
            values = right[0] * np.transpose(left)
            a_matrix[index + 1] = [transpose[0], transpose[1], transpose[2], 0, 0, 0, values[0], values[1], values[2]]
            index = index + 2
        #a_matrix[index - 1] = [0, 0, 0, 0, 0, 0, 0, 0, 1]
        #print(a_matrix)





import os, sys
import cv2
import numpy as np
from utilsImageStitching import *


def pickImages(images, keys, descriptors):
    print('pick images....')
    most_inliers = 0
    top_image = 1
    champ_H = np.zeros((3,3))

    for i in range(1, len(images)):
        print('inliers %d' %most_inliers)
        print('get matches...')
        match = getMatches(descriptors[0], descriptors[i])
        print('get ransac...')
        if len(match[0]) > 8:
            H,inliers = RANSAC(match, keys[0], keys[i])
            if inliers > most_inliers:
                print(inliers)
                most_inliers = inliers
                champ_H = H
                top_image = i
    return top_image, most_inliers, champ_H

imagePath = sys.argv[1]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE))


# Build your strategy for multi-image stitching.
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the
# number of inliers to each transformation, use this information to determine which
# pair of images should be merged first (and of these, which one should be the "source"
# and which the "destination"), merge this pair, and proceed recursively.

# YOUR CODE STARTS HERE
inliers = []
keys = []
descriptors = []
print('get keys and desc...')
for i in range(len(images)):
    keys.append(detectKeypoints(images[i]/255))
    descriptors.append(computeDescriptors(images[i], keys[i]))

most_inliers = 0
top_image1 = 0
top_image2 = 1

index1, inliers, H = pickImages(images, keys, descriptors)
image2 = images.pop(index1)
keys.pop(index1)
descriptors.pop(index1)
image1 = images.pop(0)
new_image = warpImageWithMapping(image1, image2, H)
images.insert(0, new_image)
keys.pop(0)
descriptors.pop(0)
keys.insert(0, detectKeypoints(images[0]))
descriptors.insert(0, computeDescriptors(images[0], keys[0]))


print('Putting images together...')
while len(images) > 1:
    index1,inliers,H  = pickImages(images, keys, descriptors)
    if inliers == 0:
        images.pop(index1)
    else:
        image1 = images.pop(index1)
        base_image = images.pop(0)
        keys.pop(i)
        descriptors.pop(i)
        new_image = warpImageWithMapping(base_image, image1, H)
        images.insert(0, new_image)
        keys.insert(0, detectKeypoints(images[0]))
        descriptors.insert(0,computeDescriptors(images[0],keys[0]))


imCurrent = images[0]

cv2.imshow('Panorama', imCurrent)

cv2.waitKey()

cv2.imwrite(sys.argv[2], imCurrent)

cv2.imshow('Panorama', imCurrent)

cv2.waitKey()


def pickImages(images, keys, descriptors):
    print('pick images....')
    most_inliers = 0
    image1 = 0
    image2 = 1
    champ_H = np.zeros((3,3))
    for i in range(0, len(images)):
        for j in range(1, len(images)):
            #print('inliers %d' %most_inliers)
            #print('get matches...')
            match = getMatches(descriptors[i], descriptors[j])
            print('get ransac...')
            if len(match[0]) > 8:
                H,inliers = RANSAC(match, keys[i], keys[j])
                if inliers > most_inliers:
                    print(inliers)
                    most_inliers = inliers
                    champ_H = H
                    image1 = i
                    image2 = j
    return image1, image2, champ_H




 #print('inliers %d' %most_inliers)
            #print('get matches...')
            match = getMatches(descriptors[i], descriptors[j])
            print('get matches...')
            if len(match[0]) > 8:
                H,inliers = RANSAC(match, keys[i], keys[j])
                if inliers > most_inliers:
                    print(inliers)
                    most_inliers = inliers
                    champ_H = H
                    image1 = i
                    image2 = j
    return image1, image2, champ_H



def stitch(primary_image, primary_key, primary_desc, images, keys, descriptors):
    print('sitch...')
    print('number of images %d' %len(images))
    if len(images) == 0:
        return primary_image
    else:
        champ_index = 0
        max_matches = 0
        for i in range(len(images)):
            match = getMatches(primary_desc, descriptors[i])
            if np.size(match[0]) > max_matches:
                champ_index = i
                max_matches = np.size(match[0])
                print(max_matches)
                best_match = match
        print('stitch ransac.....')
        plt.imshow(primary_image, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imshow(images[champ_index], cmap='gray', vmin=0, vmax=255)
        plt.show()
        drawMatches(primary_image, images[champ_index], matches, primary_key,keys[champ_index])
        H, inliers = RANSAC(best_match, primary_key, keys[champ_index])
        new_image = warpImageWithMapping(primary_image, images[champ_index], H)
        plt.imshow(new_image, cmap='gray', vmin=0, vmax=255)
        plt.show()
        # remove utilized image
        images.pop(champ_index)
        keys.pop(champ_index)
        descriptors.pop(champ_index)

        # recompute primary image
        primary_key = detectKeypoints(new_image)
        primary_desc = computeDescriptors(new_image, primary_key)

        cv2.imshow('Panorama', primary_image)
        cv2.waitKey()

        final = stitch(new_image, primary_key, primary_desc, images, keys, descriptors)

        return final


def pickImages(images, keys, descriptors):
    print('pick images....')
    max_matches = 0
    image1 = 0
    image2 = 1
    remove = []
    for i in range(0, len(images)-1):
        mm = 0
        for j in range(1, len(images)):
            match = getMatches(descriptors[i], descriptors[j])
            print('get matches...')
            if np.size(match[0]) > max_matches:
                max_matches = np.size(match[0])
                mm = np.size(match[0])
                best_matches = match
                image1 = i
                image2 = j
        if mm < 50:
            remove.append(i)
    return image1, image2, best_matches, remove

imagePath = sys.argv[1]

images = []
for fn in os.listdir(imagePath):
    images.append(cv2.imread(os.path.join(imagePath, fn), cv2.IMREAD_GRAYSCALE))


# Build your strategy for multi-image stitching.
# For full credit, the order of merging the images should be determined automatically.
# The basic idea is to first run RANSAC between every pair of images to determine the
# number of inliers to each transformation, use this information to determine which
# pair of images should be merged first (and of these, which one should be the "source"
# and which the "destination"), merge this pair, and proceed recursively.

# YOUR CODE STARTS HERE
inliers = []
keys = []
descriptors = []
print('get keys and desc...')
for i in range(len(images)):
    keys.append(detectKeypoints(images[i]/255))
    descriptors.append(computeDescriptors(images[i], keys[i]))

index1, index2, matches, remove = pickImages(images, keys, descriptors)
image1 = images[index1]
image2 = images[index2]


print('ransac primary....')
H, inliers = RANSAC(matches, keys[index1], keys[index2])
new_image = warpImageWithMapping(image1, image2, H)
images.pop(index1)
images.pop(index2)
keys.pop(index1)
keys.pop(index2)
descriptors.pop(index1)
descriptors.pop(index2)

new_images = []
new_keys = []
new_desc = []
for i in range(len(images)):
    if i not in remove:
        new_images.append(images[i])
        new_desc.append(descriptors[i])
        new_keys.append(keys[i])
keys = new_keys
images = new_images
descriptors = new_desc

primary_image = new_image
primary_key = detectKeypoints(primary_image)
primary_desc = computeDescriptors(primary_image, primary_key)

print('Putting images together...')
primary_image = stitch(primary_image, primary_key, primary_desc, images, keys, descriptors)

cv2.imshow('Panorama', primary_image)

cv2.waitKey()

cv2.imwrite(sys.argv[2], primary_image)




def getMatches(descriptors1, descriptors2):
    # YOUR CODE STARTS HERE
    threshold = 175
    xrange = np.shape(descriptors1)[0]
    yrange = np.shape(descriptors2)[0]
    index1 = []
    index2 = []
    values = []
    ibest = 0
    # iterate through all descriptors and find matches within a range
    for i in range(0,xrange):
        ibest = -1
        bestvalue = float('inf')
        for j in range(0, yrange):
            value = np.sqrt(np.sum((descriptors1[i,:] - descriptors2[j,:]) ** 2))
            if value <= bestvalue and value > 0:
                bestvalue = value
                ibest = j
                #index1.append(i)
                #index2.append(j)
                values.append(values)
        if ibest != -1:
            index1.append(i)
            index2.append(ibest)

    # eliminate matches if too many
    if len(values) > 150:
        while len(values) > 150:
            remove_index = values.index(max(values))
            index1.pop(remove_index)
            index2.pop(remove_index)
            values.pop(remove_index)

    return index1, index2


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
