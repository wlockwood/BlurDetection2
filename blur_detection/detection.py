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

    # print(f"Resizing image from {x0}x{y0} ({start_pix/1E6:.1f}MP) by {resize_ratio:.3f} in each dimension"
    #             f" to {x1:.0f}x{y1:.0f} ({end_pix/1E6:.1f}MP)")
    return cv2.resize(image, (x1, y1), interpolation=cv2.INTER_CUBIC)


def estimate_blur(image: numpy.array):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = numpy.var(blur_map)
    return blur_map, score


def pretty_blur_map(blur_map: numpy.array, sigma: int = 5, min_abs: float = 0.5):
    abs_image = numpy.abs(blur_map).astype(numpy.float32)
    abs_image[abs_image < min_abs] = min_abs

    abs_image = numpy.log(abs_image)
    cv2.blur(abs_image, (sigma, sigma))
    return cv2.medianBlur(abs_image, sigma)
