import numpy as np
import cv2

# Common file impementing some complex transforms like:
#   gamma adjustment, summing/blending images, blur filter etc.
# (as per Q4 & Q8 on the FAQ these are self-written & do not use complex opencv methods)

def adjust_gamma(image, gamma=1.0):
  invGamma = 1.0 / gamma
  image = image / 255 
  image = image ** invGamma
  image = np.floor(255 * image)
  return image.astype(np.uint8)

def kernel_filter(image, scalar, kernel):
  rows = image.shape[0]
  cols = image.shape[1]
  ksize = kernel.shape[0]
  lim = (ksize - 1) // 2

  out = np.zeros(image.shape, np.uint8)

  image = np.pad(image, ksize - 1, mode='edge')

  for i in range(rows):
    ik = i + ksize - 1
    for j in range(cols):
      jk = j + ksize - 1
      roi = image[ik-lim:ik+lim+1,jk-lim:jk+lim+1]
      out[i][j] = scalar * np.sum(roi * kernel)

  return out

def filter_image(image, scalar, kernel):
  if(len(image.shape) == 3):
    channels = cv2.split(image)
  else:
    channels = (image)

  for i in range(len(channels)):
    channels[i] = kernel_filter(channels[i], scalar, kernel)

  return cv2.merge(channels)

def box_blur(image, ksize = 3):
  return filter_image(image, 1 /ksize ** 2, np.ones((ksize, ksize), np.uint8))

def motion_blur_horizontal(image, ksize = 3):
  kernel = np.zeros((ksize, ksize), dtype=np.uint8)
  kernel[ksize // 2] = [1] * ksize
  return filter_image(image, 1 / ksize, kernel)

def add_weighted(fg, alpha, bg, beta):
  out = np.zeros(bg.shape, np.uint8)
  for i in range(bg.shape[0]):
    for j in range(bg.shape[1]):
      a = fg[i][j] * alpha
      b = bg[i][j] * beta
      out[i][j] = a + np.minimum(255 - a, b)
  
  return out.astype(np.uint8)
