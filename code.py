import cv2
import numpy as np
from scipy import interpolate


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    image = image / 255
    image = image ** invGamma
    image = np.floor(255 * image)
    return image.astype(np.uint8)


def box_blur(image, ksize=3):
    return cv2.filter2D(image, -1, np.ones((ksize, ksize)) / (ksize ** 2))


def motion_blur_horizontal(image, ksize=3):
    kernel = np.zeros((ksize, ksize), dtype=np.uint8)
    kernel[ksize // 2] = [1] * ksize
    return cv2.filter2D(image, -1, kernel / ksize)


def add_weighted(fg, alpha, bg, beta):
    out = np.zeros(bg.shape, np.uint8)
    for i in range(bg.shape[1]):
        for j in range(bg.shape[0]):
            a = fg[i][j] * alpha
            b = bg[i][j] * beta
            out[i][j] = a + np.minimum(255 - a, b)

    return out.astype(np.uint8)


# Expects a color image and color mask
def problem1(image, mask, darkening_coeff=0.6, blending_coeff=0.4, blur=7, rainbow=False, rainbow_stretch=1.1):
    # See https://www.desmos.com/calculator/j9k8tz4nml for these functions graphed
    # w = width of row, a = max amplitude of warp, x = position of pixel in row (left to right)
    def blue_adjust(w, a, x):
        return a / w * x

    def red_adjust(w, a, x):
        return -a / w * (x - w)

    def green_adjust(w, a, x):
        return -abs((2 * a) / w * x - a) + a

    def rainbowify(im, b, g, r):
        rows, cols, channels = image.shape
        for i in range(rows):
            row = []

            for j in range(cols):
                if np.all(im[i][j]):
                    row.append((i, j))

            width = len(row)
            if width == 0:
                continue

            for j in range(width):
                n, m = row[j]
                im[n][n][0] *= blue_adjust(width, b, j)
                im[n][m][1] *= green_adjust(width, g, j)
                im[n][m][2] *= red_adjust(width, r, j)

        return im

    # Invert mask + threshold it
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY_INV)

    # Darken image
    darkened = adjust_gamma(image, darkening_coeff)

    # Mask out the "bright" segment of our original segment with our mask
    fg = cv2.bitwise_and(image, image, mask=mask)

    if rainbow:
        fg = rainbowify(fg, rainbow_stretch, rainbow_stretch, rainbow_stretch)

    fg = box_blur(fg, blur)

    # Blend the bright segment onto the darkened base image
    return add_weighted(fg, blending_coeff, darkened, 1)


# Expects a color image
def problem2(image, noise1=50, noise2=100, blending_coeff=0.06, blur=13, colored=False):
    def pencil_noise(var):
        mean = 0
        sigma = var ** 0.5
        noise = np.random.normal(mean, sigma, image.shape)
        noise = motion_blur_horizontal(noise, blur)
        noise = noise.astype(np.uint8)
        return noise

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise1 = pencil_noise(noise1)
    if colored:
        noise2 = pencil_noise(noise2)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        b, g, r = cv2.split(image)
        r = add_weighted(noise1, blending_coeff, r, 1 - blending_coeff)
        r = adjust_gamma(r, 1.5)
        b = add_weighted(noise2, blending_coeff, b, 1 - blending_coeff)
        b = adjust_gamma(b, 1.8)
        out = cv2.merge((b, g, r))
    else:
        out = add_weighted(noise1, blending_coeff, image, 1 - blending_coeff)

    return out


# Expects a color image
def problem3(image, blur=3):
    def curve(channel, values):
        xs = [i[0] for i in values]
        ys = [i[1] for i in values]

        tck = interpolate.splrep(xs, ys, s=0, k=2)
        rng = np.arange(0, 256)
        y_lut = interpolate.splev(rng, tck).astype(np.uint8)

        return cv2.LUT(channel, y_lut)

    out = box_blur(image, blur)
    b, g, r = cv2.split(out)
    b = curve(b, [[0, 0], [108, 90], [255, 255]])
    g = curve(g, [[0, 0], [113, 104], [255, 255]])
    r = curve(r, [[0, 0], [95, 91], [255, 255]])
    out = cv2.merge((b, g, r))

    return curve(out, [[0, 0], [178, 133], [255, 255]])


# Expects a color image
def problem4(image, warp_angle=np.pi / 2, warp_radius=120, bilinear_interp=True):
    # Expects cartesian coordinates in center-origin format (x, y)
    # Returns polar coordinates in form(R, theta)
    def polar(pt):
        x, y = pt
        theta = np.arctan2(y, x)
        return [np.sqrt(x ** 2 + y ** 2), theta]

    # Expects polar coordinates in form(R, theta)
    # Returns coordinates in center origin format (x, y)
    def cartesian(ppt):
        r, theta = ppt
        return r * np.cos(theta), r * np.sin(theta)

    # Expects cartesian coordinates in center-origin format (x, y)
    # Returns cartesian coordinates in center-origin format (x, y)
    def warp(pt):
        ppt = polar(pt)
        s = np.maximum(0, (warp_radius - ppt[0]) / warp_radius)
        ppt[1] += s * warp_angle
        return cartesian(ppt)

    # Simple nearest neighbour by rounding values towards zero then converting to int
    def nearest_neighbour(pt):
        try:
            return image[int(np.fix(pt[0])), int(np.fix(pt[1]))]
        except IndexError:
            return 0

    def bilinear(pt):
        y, x = pt
        return [fb(x, y), fg(x, y), fr(x, y)]

    # Converts cartesian coords in center-origin format (x, y) -> cartesian coords in top-left-origin format (x, y)
    def originate(pt):
        return pt[0] + Cy, pt[1] + Cx

    H, W = image.shape[:2]
    Cx, Cy = W // 2, H // 2
    R, C = np.arange(-Cy, Cy), np.arange(-Cx, Cx)

    warped = np.zeros(image.shape, np.uint8)

    if bilinear_interp:
        cB, cG, cR = cv2.split(image)
        fb = interpolate.interp2d(C + Cy, R + Cx, cB)
        fg = interpolate.interp2d(C + Cy, R + Cx, cG)
        fr = interpolate.interp2d(C + Cy, R + Cx, cR)

    # For each pixel in the output usually our warp settings determine the pixel to use from the input
    #  then interpolate it back to a precise pixel in the input
    for i in R:
        for j in C:
            point = warp((i, j))
            point = originate(point)

            if bilinear_interp:
                p = bilinear(point)
            else:
                p = nearest_neighbour(point)

            warped[(i, j)] = p

    shifted = np.zeros(image.shape, np.uint8)

    # Deals with odd-dimensioned pictures when quadrant swapping
    if W % 2 != 0:
        Cx += 1

    if H % 2 != 0:
        Cy += 1

    # swap quads 1, 3
    shifted[H - Cy:, W - Cx:] = warped[0:Cy, 0:Cx]
    shifted[0:Cy, 0:Cx] = warped[H - Cy:, W - Cx:]

    # swap quads 2, 4
    shifted[0:Cy, W - Cx:] = warped[H - Cy:, 0:Cx]
    shifted[H - Cy:, 0:Cx] = warped[0:Cy, W - Cx:]
    return shifted


face1 = cv2.imread("face1.jpg")
#face2 = cv2.imread("face2.jpg")

mask1 = cv2.imread("mask1.jpg")
#mask2 = cv2.imread("mask2.jpg")

face1 = problem2(face1)
#face2 = problem1(face2, mask1, darkening_coeff=0.5, rainbow=True)

cv2.imshow("face1-ll.jpg", face1)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite("face2-ll.jpg", face2)
