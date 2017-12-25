import cv2
import numpy as np

import helper
import util

cam_cal = np.load('./cam_cal.npz')


def binary_thres(img, lower_pct=97, upper_pct=100, lower=None, upper=None):
    assert len(img.shape) == 2 or img.shape[0] == 1

    if lower is None:
        lower = np.percentile(img, lower_pct)
    if upper is None:
        upper = np.percentile(img, upper_pct)

    print(lower, upper)

    binary = np.zeros_like(img)
    binary[(img >= lower) & (img <= upper)] = 1

    return binary


def _scale_img(img):
    abs_img = np.absolute(img)
    scaled = np.uint8(255 * abs_img / np.max(abs_img))
    return scaled


def sobel_thres(img, sobel_kernel=3):
    assert len(img.shape) == 2 or img.shape[-1] == 1

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    gradx = binary_thres(sobelx, lower=0, upper=255)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    grady = binary_thres(sobely, lower=0, upper=255)

    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    mag_bin = binary_thres(gradmag, lower=190, upper=255)

    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_margin = np.pi / 10
    lower_dir = np.pi / 2 - dir_margin
    upper_dir = np.pi / 2 + dir_margin
    condition = (absgraddir >= lower_dir) & (absgraddir < upper_dir)

    dir_bin = np.zeros_like(img)
    dir_bin[condition] = 1

    combined_bin = np.zeros_like(img)
    condition = ((mag_bin == 1) | (dir_bin == 1))
    combined_bin[condition] = 255

    return np.dstack([combined_bin] * 3)


def edge_detection(rgb_img, s_only=False):
    R = rgb_img[:, :, 0]
    r_binary = binary_thres(R)
    r_edge = sobel_thres(r_binary)  # helper.canny(img)
    return r_edge

    hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    s_binary = binary_thres(S)
    s_edge = sobel_thres(s_binary)

    final = s_edge

    if not s_only:
        final = final // 2 + r_edge // 2

    return np.dstack([final] * 3)


def undistort_img(img):
    mtx = cam_cal['mtx']
    dist = cam_cal['dist']

    return cv2.undistort(img, mtx, dist, None, None)


def bird_eye_view(img, src_corners, w, h, offset=50):
    print('W', w)
    print('H', h)
    dst_corners = np.array([(offset, offset), (w - offset, offset),
                            (w - offset, h - offset),
                            (offset, h - offset)]).astype('float32')
    M = cv2.getPerspectiveTransform(src_corners, dst_corners)
    dst = cv2.warpPerspective(img, M, (w, h))

    return dst
