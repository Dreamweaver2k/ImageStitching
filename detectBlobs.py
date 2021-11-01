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
# Output:
#   blobs          - n x 4 array with blob in each row in (x, y, radius, score)
#
def DetectBlobs(
    im,
    sigma = 3,
    num_octaves = 3,
    num_intervals = 12,
    threshold = 1e-4
    ):

    # Convert image to grayscale and convert it to double [0 1].
    if len(im.shape) > 2:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)/255

    # YOUR CODE STARTS HERE
    # Compute filters to be applied later
    size = int(sigma * 6)
    full_im = np.zeros(np.shape(im))
    inter_im = np.zeros(np.shape(im))
    ref_radius = np.zeros(np.shape(im))
    inter_radius = np.zeros(np.shape(im))
    radii = np.zeros(num_intervals)

    # calculate filters for each value k
    filters = []
    for i in range(0, num_intervals):
        k = pow(2, (i / num_intervals))
        filter_size = int(k * size)

        if filter_size % 2 == 0:
            filter_size = filter_size + 1
        filters.append(laplacian(k * sigma, filter_size))
        radii[i] = math.sqrt(2) * k * (sigma)


#####################################################################
    # Apply filters for all intervals and octaves
    for i in range(0, num_octaves):
        for j in range(0, num_intervals):
            k = pow(2, j/num_intervals)

            # downsize if necessary
            resize_factor = 1 / (pow(2, i))
            height, width = np.shape(im)
            height = int(height * resize_factor)
            width = int(width * resize_factor)
            dsize = (width, height)
            temp_im = cv2.resize(im, dsize)

            # convolve absolute power and normalize
            temp_im = ndimage.convolve(temp_im, filters[j],mode='constant')

            temp_im = nms(temp_im, sigma * k * 2**i)

            # Resize to larger image
            height, width = np.shape(im)
            dsize = (width, height)
            temp_im = cv2.resize(temp_im, dsize)


            # magnitude of response
            temp_im = np.square(temp_im)

            # find max value at all positions
            xrange, yrange = np.shape(temp_im)
            for x in range(xrange):
                for y in range(yrange):
                    if temp_im[x][y] > inter_im[x][y]:
                        inter_im[x][y] = temp_im[x][y]
                        inter_radius[x][y] = radii[j] * pow(2, i)

        inter_im = nms(inter_im, sigma * pow(2,i))

        # find max value at all positions
        xrange,yrange = np.shape(inter_im)
        for x in range(xrange):
            for y in range(yrange):
                if inter_im[x][y] > full_im[x][y]:
                    full_im[x][y] = inter_im[x][y]
                    ref_radius[x][y] = inter_radius[x][y]
    # another filter for the layers of filters
    full_im = nms(full_im, 4)

    blobs = np.array([[0, 0, 0, 0]])

    xrange, yrange = np.shape(full_im)
    for x in range(xrange):
        for y in range(yrange):
            if full_im[x][y] > threshold:
                new_blob = np.array((x, y, ref_radius[x][y], full_im[x][y]))
                blobs = np.vstack((blobs, new_blob))
    return blobs[1:]

def nms(im, sigma):
    filter = ndimage.maximum_filter(im, size=int(sigma*math.sqrt(2)))
    filter = (im - filter)
    xrange, yrange = np.shape(im)
    for x in range(0, xrange):
        for y in range(0, yrange):
            if filter[x][y] != 0:
                im[x][y] = 0
    return im


# Computes a laplacian filter given inputs sigma and a size for the filter
def laplacian(sigma, size):
    laplacian_filter = np.zeros((size,size))
    center = int(size / 2)
    g = gaussian(sigma, size)
    for x in range(0, size):
        for y in range(0, size):
            laplacian_filter[x][y] = (pow(x - center, 2) + pow(y - center, 2) - 2*sigma**2) / (math.pow(sigma, 4))

    laplacian_filter = laplacian_filter * g
    laplacian_filter = laplacian_filter - np.mean(laplacian_filter)
    return laplacian_filter


#computes a gaussian filter given inputs sigma and size of the filter
def gaussian(sigma, size):
    gaussian_filter = np.zeros((size, size))
    center = int(size / 2)
    for x in range(0, size):
        for y in range(0, size):
            gaussian_filter[x][y] = (1/(2*math.pi*sigma**2))*math.exp(-1*((x-center)**2 + (y-center)**2)/(2*sigma**2))
    gaussian_filter = gaussian_filter/np.sum(gaussian_filter)
    return gaussian_filter
