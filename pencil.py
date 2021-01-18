import cv2
import numpy as np
import common_utils as utils

# Params
IMAGE = "face1.jpg"
NOISE1 = 100
NOISE2 = 0.1
BLENDING_COEFF = 0.10
BLUR = 3
COLORED = True


def pencil_noise(var):
    mean = 0
    sigma = var ** 0.5
    noise = np.random.normal(mean, sigma, img.shape)
    noise = utils.motion_blur_horizontal(noise, BLUR)
    cv2.imshow("noise", noise)
    return noise


img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)
noise1 = pencil_noise(NOISE1)

if COLORED:
    noise2 = pencil_noise(NOISE2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(img)
    r = utils.add_weighted(noise1, BLENDING_COEFF, r, 1 - BLENDING_COEFF)
    r = utils.adjust_gamma(r, 1.5)
    b = utils.add_weighted(noise2, BLENDING_COEFF, b, 1 - BLENDING_COEFF)
    b = utils.adjust_gamma(b, 1.8)
    out = cv2.merge((b, g, r))
else:
    out = utils.add_weighted(noise1, BLENDING_COEFF, img, 1 - BLENDING_COEFF)

cv2.imshow("in", img)
cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
