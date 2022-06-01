#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy
from math import sqrt


def fix_image_size(image: numpy.array, expected_pixels: float = 2e6):
    x0, y0 = image.shape[1], image.shape[0]
    start_pix = x0 * y0
    aspect_ratio = y0 / x0

    x1 = round(sqrt(expected_pixels / aspect_ratio))
    y1 = round(x1 * aspect_ratio)
    end_pix = x1 * y1
    resize_ratio = x1 / x0

    #print(f"Resizing image from {x0}x{y0} ({start_pix/1E6:.1f}MP) by {resize_ratio:.3f} in each dimension"
                 # f" to {x1:.0f}x{y1:.0f} ({end_pix/1E6:.1f}MP)")
    return cv2.resize(image, (x1, y1), interpolation=cv2.INTER_CUBIC)


def estimate_blur(image: numpy.array, just_score:bool=False):
    """
    Evaluates an image for blurriness.
    Returns score that correlates smaller values with blurriness.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    if just_score:
        return score
    else:
        return blur_map, score


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)


def detect_blur_fft(image, size=60, thresh=10, vis=False):
    """
    Evaluates an image for blurriness.
    Returns score that correlates smaller values with blurriness.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # copied from https://pyimagesearch.com/2020/06/15/opencv-fast-fourier-transform-fft-for-blur-detection-in-images-and-video-streams/
    # grab the dimensions of the image and use the dimensions to
    # derive the center (x, y)-coordinates
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))

    # compute the FFT to find the frequency transform, then shift
    # the zero frequency component (i.e., DC component located at
    # the top-left corner) to the center where it will be more
    # easy to analyze
    fft = numpy.fft.fft2(image)
    fftShift = numpy.fft.fftshift(fft)

    # check to see if we are visualizing our output
    if vis:
        # compute the magnitude spectrum of the transform
        magnitude = 20 * numpy.log(numpy.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("Input")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        plt.show()

    # zero-out the center of the FFT shift (i.e., remove low
    # frequencies), apply the inverse shift such that the DC
    # component once again becomes the top-left, and then apply
    # the inverse FFT
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = numpy.fft.ifftshift(fftShift)
    recon = numpy.fft.ifft2(fftShift)

    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * numpy.log(numpy.abs(recon))
    mean = numpy.mean(magnitude)

    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    return mean
