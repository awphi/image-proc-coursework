import cv2
import numpy as np
from scipy import interpolate
import common_utils as utils


# FILTER NAME: WARM NOIR

def curve(channel, values):
    xs = [i[0] for i in values]
    ys = [i[1] for i in values]

    tck = interpolate.splrep(xs, ys, s=0, k=2)
    rng = np.arange(0, 256)
    y_lut = interpolate.splev(rng, tck).astype(np.uint8)

    return cv2.LUT(channel, y_lut)


# Params
IMAGE = "face1.jpg"
BLUR = 3

img = cv2.imread(IMAGE)

out = utils.box_blur(img, BLUR)
b, g, r = cv2.split(out)
b = curve(b, [[0, 0], [108, 90], [255, 255]])
g = curve(g, [[0, 0], [113, 104], [255, 255]])
r = curve(r, [[0, 0], [95, 91], [255, 255]])
out = cv2.merge((b, g, r))
out = curve(out, [[0, 0], [178, 133], [255, 255]])

cv2.imshow("in", img)
cv2.imshow("out", out)
cv2.waitKey(0)
cv2.destroyAllWindows()
