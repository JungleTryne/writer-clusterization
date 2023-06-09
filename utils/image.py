# Taken from https://github.com/QuwsarOhi/BanglaWriting/blob/main/writingMod.py

import cv2
import numpy as np


def convert_image(img: np.ndarray) -> np.ndarray:
    img = np.array(img)
    img = 255 - img
    img = img / 255

    return img


# Taken from https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
