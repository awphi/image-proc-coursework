import cv2
import common_utils as utils

# Params
IMAGE = "face1.jpg"
NOISE1 = "noise1.jpg"
NOISE2 = "noise2.jpg"
BLENDING_COEFF = 0.15
BLUR = 5
COLORED = False


def pencil_noise(st):
    # TODO replace with generated noise!
    noise = cv2.imread(st, cv2.IMREAD_GRAYSCALE)
    noise = utils.motion_blur_horizontal(noise, BLUR)
    return noise


noise1 = pencil_noise(NOISE1)

img = cv2.imread(IMAGE, cv2.IMREAD_GRAYSCALE)

if COLORED:
    noise2 = pencil_noise(NOISE2)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    b, g, r = cv2.split(img)
    r = utils.add_weighted(noise1, BLENDING_COEFF, r, 1 - BLENDING_COEFF)
    r = utils.adjust_gamma(r, 1.5)
    b = utils.add_weighted(noise2, BLENDING_COEFF, b, 1 - BLENDING_COEFF)
    b = utils.adjust_gamma(b, 0.6)
    out = cv2.merge((b, g, r))
else:
    out = utils.add_weighted(noise1, BLENDING_COEFF, img, 1 - BLENDING_COEFF)

cv2.imshow("in", img)
cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
