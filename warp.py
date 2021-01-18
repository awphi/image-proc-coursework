import cv2
import numpy as np

# Params
IMAGE = "face1.jpg"
WARP_ANGLE = np.pi * 2
WARP_RADIUS = 150
BILINEAR_INTERP = False


# Expects cartesian coordinates in center-origin format (x, y)
# Returns polar coordinates in form(r, theta)
def polar(pt):
    x, y = pt
    theta = np.arctan2(y, x)
    return [np.sqrt(x ** 2 + y ** 2), theta]


# Expects polar coordinates in form(r, theta)
# Returns coordinates in center origin format (x, y)
def cartesian(ppt):
    r, theta = ppt
    return r * np.cos(theta), r * np.sin(theta)


# Expects cartesian coordinates in center-origin format (x, y)
# Returns cartesian coordinates in center-origin format (x, y)
def warp(pt):
    ppt = polar(pt)
    s = np.maximum(0, (WARP_RADIUS - ppt[0]) / WARP_RADIUS)
    ppt[1] += s * WARP_ANGLE
    return cartesian(ppt)


# Simple nearest neighbour by rounding values towards zero then converting to int
def nearest_neighbour(pt):
    return int(np.fix(pt[0])), int(np.fix(pt[1]))


# Converts cartesian coords in center-origin format (x, y) -> cartesian coords in top-left-origin format (x, y)
def originate(pt):
    return pt[0] + Cy, pt[1] + Cx


img = cv2.imread(IMAGE)
H, W = img.shape[:2]
Cx, Cy = W // 2, H // 2
R, C = np.arange(-Cy, Cy), np.arange(-Cx, Cx)

warped = np.zeros(img.shape, np.uint8)

# For each pixel in the output usually our warp settings determine the pixel to use from the input
#  then interpolate it back to a precise pixel in the input
for i in R:
    for j in C:
        point = warp((i, j))
        point = originate(point)
        if BILINEAR_INTERP:
            # TODO bilinear interpolation
            a = 1
        else:
            point = nearest_neighbour(point)

        # Deals with trying to access a pixel outside the source image, just set it to black
        try:
            p = img[point]
        except IndexError:
            p = 0
        warped[(i, j)] = p

shifted = np.zeros(img.shape, np.uint8)

# Deals with odd-dimensioned pictures when quadrant swapping
if W % 2 != 0:
    Cx += 1

if H % 2 != 0:
    Cy += 1


# swap q1 and q3
shifted[H-Cy:, W-Cx:] = warped[0:Cy, 0:Cx]
shifted[0:Cy, 0:Cx] = warped[H - Cy:, W - Cx:]

# swap q2 and q4
shifted[0:Cy, W-Cx:] = warped[H - Cy:, 0:Cx]
shifted[H-Cy:, 0:Cx] = warped[0:Cy, W - Cx:]

cv2.imshow("in", img)
cv2.imshow("out", shifted)

cv2.waitKey(0)
cv2.destroyAllWindows()
