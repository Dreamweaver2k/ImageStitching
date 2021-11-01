import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utilsImageStitching import *


# Recursively stitches a set of images together given the primary image, the set of images, and all their respective
# descriptors and keys.
def stitch(primary_image, primary_key, primary_desc, images, keys, descriptors):
    if len(images) == 0:
        return primary_image
    else:
        champ_index = 0
        max_matches = 0
        for i in range(len(images)):
            match = getMatches(descriptors[i], primary_desc)
            if np.size(match) > max_matches:
                champ_index = i
                max_matches = np.size(match[0])
                best_match = match
        H, inliers = RANSAC(best_match, keys[champ_index],primary_key)
        new_image = warpImageWithMapping(images[champ_index], primary_image, H)

        # remove utilized image
        images.pop(champ_index)
        keys.pop(champ_index)
        descriptors.pop(champ_index)

        if len(images) > 0:
            # recompute primary image
            primary_key = detectKeypoints(new_image / 255)
            primary_desc = computeDescriptors(new_image, primary_key)
            final = stitch(new_image, primary_key, primary_desc, images, keys, descriptors)
        else:
            return new_image
        return final


# Returns the indices of what images should be merged first given the images, their keys, and their descriptors.
# Also returns a list of images to be filtered out from the list for being unlike the other images.
def pickImages(images, keys, descriptors):
    max_inliers = 0
    image1 = 0
    image2 = 1
    remove = []
    for i in range(0, len(images)):
        mm = 0
        for j in range(0, len(images)):
            if j != i:
                match = getMatches(descriptors[i], descriptors[j])
                if np.size(match[0]) > 4:
                    H, inliers = RANSAC(match, keys[i], keys[j])
                else:
                    inliers = 0
                if inliers > mm:
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_H = H
                        image1 = i
                        image2 = j
                    mm = inliers
        if mm < 4:
            remove.append(i)
    return image1, image2, best_H, remove


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
for i in range(len(images)):
    keys.append(detectKeypoints(images[i]/255))
    descriptors.append(computeDescriptors(images[i], keys[i]))

index1, index2, H, remove = pickImages(images, keys, descriptors)
image1 = images[index1]
image2 = images[index2]


new_image = warpImageWithMapping(image1, image2, H)
remove.append(index1)
remove.append(index2)
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
primary_key = detectKeypoints(primary_image/255)
primary_desc = computeDescriptors(primary_image, primary_key)

primary_image = stitch(primary_image, primary_key, primary_desc, images, keys, descriptors)

cv2.imshow('Panorama', primary_image)

cv2.waitKey()

cv2.imwrite(sys.argv[2], primary_image)




