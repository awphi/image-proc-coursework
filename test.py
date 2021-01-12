import cv2
import numpy as np
import common_utils as utils

IMAGE = "face1.jpg"

img = cv2.imread(IMAGE)
img = utils.motion_blur_horizontal(img, 19)

cv2.imshow("img", img)

cv2.waitKey(0)
cv2.destroyAllWindows()