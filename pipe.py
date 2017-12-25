#!/usr/bin/env python
import os
import glob
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

import helper
import helper2
import util
from importlib import reload

helper = reload(helper)
util = reload(util)
_state_cache = defaultdict(dict)

# Checkerboard pattern corners
NX = 9
NY = 6


def cam_calibration(viz=False):
    # Ref: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html
    objp = np.zeros((NX * NY, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)

    cam_imgs = [cv2.imread(f) for f in glob.glob('./camera_cal/*')]
    cam_imgs2 = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in cam_imgs]

    ret_corners = [
        cv2.findChessboardCorners(img, (NX, NY), None) for img in cam_imgs2
    ]

    objpoints = []
    imgpoints = []
    for idx, (ret, corners) in enumerate(ret_corners):
        if not ret:
            continue

        objpoints.append(objp)
        imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, cam_imgs2[0].shape[::-1], None, None)

    np.savez_compressed('cam_cal', mtx=mtx, dist=dist)

    if viz:
        idx = 0
        ret, corners = ret_corners[idx]
        img = cv2.drawChessboardCorners(cam_imgs[idx], (NX, NY), corners, ret)
        cv2.imwrite('sample.png', img)

        img = cv2.undistort(img, mtx, dist, None, None)
        cv2.imwrite('undist.png', img)

    return mtx, dist


def detect_lane(rgb_img, state_id=None):
    '''Detects and draw lane overlay; returns img as np array'''
    global _state_cache

    h, w, chan = rgb_img.shape

    img = helper.grayscale(rgb_img)
    img = helper.gaussian_blur(img, 5)

    # mask other regions
    region = np.array([
        # middle center
        (int(w / 2), int(h / 2)),
        # bottom left
        (int(w * 0.1), int(h * 0.90)),
        # bottom right
        (int(w * 0.9), int(h * 0.90)),
    ])
    img = helper.canny(img, 30, 150)
    img = helper.region_of_interest(img, [region])

    state = _state_cache[state_id] if state_id else None
    img = helper.hough_lines(
        img,
        rho=2,
        theta=np.pi / 180,
        threshold=64,
        min_line_len=50,
        max_line_gap=40,
        state=state)
    img = helper.weighted_img(img, rgb_img)

    return img


def detect_lane2(rgb_img, state_id=None):
    '''
    The goals / steps of this project are the following:
        x Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        x Apply a distortion correction to raw images.
        x Use color transforms, gradients, etc., to create a thresholded binary image.
        x Apply a perspective transform to rectify binary image ("birds-eye view").
        Detect lane pixels and fit to find the lane boundary.
        Determine the curvature of the lane and vehicle position with respect to center.
        Warp the detected lane boundaries back onto the original image.
        Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    '''
    global _state_cache

    h, w, chan = rgb_img.shape
    region = np.array([
        # top left
        (int(w / 2) - 100, int(h / 2) + 100),
        # top right
        (int(w / 2) + 100, int(h / 2) + 100),
        # bottom right
        (int(w * 0.8), int(h * 0.85)),
        # bottom left
        (int(w * 0.2), int(h * 0.85)),
    ])

    img = helper2.undistort_img(rgb_img)
    img = helper.gaussian_blur(img, 5)
    img = helper2.edge_detection(img)
    img = helper.region_of_interest(img, [region])
    img = helper2.bird_eye_view(img, region.astype('float32'), w, h)

    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    plt.plot(histogram)
    plt.show()


    return img


def process_image(output_root='./output_images',
                  img_root='test_images',
                  debug=True):
    test_imgs = os.listdir(img_root)
    if debug:
        test_imgs = test_imgs[:3]

    for path in test_imgs:
        img = mpimg.imread(os.path.join(img_root, path))
        img = detect_lane2(img)

        cv2.imwrite(
            os.path.join(output_root, path),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    import fire

    fire.Fire({
        'proc': process_image,
        'cam': cam_calibration,
    })
