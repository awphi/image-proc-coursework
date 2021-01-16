import cv2
import numpy as np
import common_utils as utils

# Params
IMAGE = "face1.jpg"
MASK = "mask1.jpg"
DARKENING_COEFF = 0.6
BLENDING_COEFF = 0.4
RAINBOW = True

# Extra params
RAINBOW_STRETCH = 1.1
BLUR = 7


# See https://www.desmos.com/calculator/j9k8tz4nml for these functions graphed
# w = width of row, a = max amplitude of warp, x = position of pixel in row (left to right)
def blue_adjust(w, a, x):
    return a / w * x


def red_adjust(w, a, x):
    return -a / w * (x - w)


def green_adjust(w, a, x):
    return -abs((2 * a) / w * x - a) + a


def rainbow(image, b, g, r):
    rows, cols, channels = img.shape
    for i in range(rows):
        row = []

        for j in range(cols):
            if np.all(image[i][j]):
                row.append((i, j))

        width = len(row)
        if width == 0:
            continue

        for j in range(width):
            n, m = row[j]
            image[n][n][0] *= blue_adjust(width, b, j)
            image[n][m][1] *= green_adjust(width, g, j)
            image[n][m][2] *= red_adjust(width, r, j)

    return image


# Invert mask + threshold it
mask_grey = cv2.imread(MASK, cv2.IMREAD_GRAYSCALE)
ret, mask = cv2.threshold(mask_grey, 200, 255, cv2.THRESH_BINARY_INV)

img = cv2.imread(IMAGE)

# Darken image
darkened = utils.adjust_gamma(img, DARKENING_COEFF)

# Mask out the "bright" segment of our original segment with our mask
fg = cv2.bitwise_and(img, img, mask=mask)

if RAINBOW:
    fg = rainbow(fg, RAINBOW_STRETCH, RAINBOW_STRETCH, RAINBOW_STRETCH)

fg = utils.box_blur(fg, BLUR)

# Blend the bright segment onto the darkened base image
out = utils.add_weighted(fg, BLENDING_COEFF, darkened, 1)

cv2.imshow("in", img)
cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
