import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cv2 import GaussianBlur, bilateralFilter, filter2D
import numpy as np

def show_image(image):
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    
def MLoG(img, kernelSize):
    #Laplace of Gaussian
    gaussianImg = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0, 0)
    laplaceImg = cv2.Laplacian(gaussianImg, cv2.CV_16S)
    laplaceImg[laplaceImg < 0] = 0
    laplaceImg = cv2.convertScaleAbs(laplaceImg)
    
    ret,threshold_img = cv2.threshold(laplaceImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #binarize otsu
    return threshold_img

def CCC(img, kernelSize):
    #Connected Component Criteria
    #KernelSize Can only be odd numbers
    
    mlog1 = MLoG(img, kernelSize)
    mlog2 = MLoG(img, kernelSize-2)
    #Ilog laplace of gaussian

    #Number of connected Components NCC
    ncc1, _ = cv2.connectedComponents(mlog1)
    ncc2, _ = cv2.connectedComponents(mlog2)

    return (1 - (ncc1 / ncc2))

def STC(img, kernelSize):
    #Stop Criteria
    #KernelSize can only be odd numbers
    
    mlog1 = MLoG(img, kernelSize)
    mlog2 = MLoG(img, kernelSize - 2)
    mlog3 = MLoG(img, 1)
    
    ncc1, _ = cv2.connectedComponents(mlog1)
    ncc2, _ = cv2.connectedComponents(mlog2)
    ncc3, _ = cv2.connectedComponents(mlog3)
    return ((ncc1/ncc3) * np.abs(ncc2 - ncc1))

def ScreenToneRemoval(img, stcThreshold=1, beta=0.8):
    '''
    Return: remoovalMask, iLog, ibase
    '''
    maxCCC = 0
    iLog = 1
    i = 3
    while STC(img, i) > stcThreshold:
        if (CCC(img, i) > (beta * maxCCC)):
            iLog = i
            print("i: ",iLog)
            maxCCC = max(maxCCC, CCC(img, i))
            
        i = i + 2
        
    iLog = iLog + 4 #ilog
    
    if (int(i/2))%2 == 0:
        ibase = min(int(i/2) + 1, iLog)
    else:
        ibase = min(int(i/2), iLog)
        
    M_logI = MLoG(img, iLog)
    M_logBase = MLoG(img, ibase)
    Mask = cv2.bitwise_and(M_logI, M_logBase)
    
    return iLog, ibase, Mask

def DifferenceOfGaussian(img, kernelSize):
    firstGaussian = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0, 0)
    secondGaussian = cv2.GaussianBlur(img, (kernelSize-2, kernelSize-2), 0, 0)
    
    return secondGaussian - firstGaussian

#to the sharpness of the screentones causing them
# to be confused with edge lines

# Gaussian blue with variable kernel size, aka more or less blurring
def blur(img, blur_amount=5):
    if(blur_amount == 7):
        dst2 = GaussianBlur(img,(7,7),0)
        dst = bilateralFilter(dst2, 7, 80, 80)
    else:
        dst2 = GaussianBlur(img,(5,5),0)
        dst = bilateralFilter(dst2, 7, 10 * blur_amount, 80)
    return dst

# Laplacian filter for sharpening. Only want to do runs of 3x3 kernels to avoid oversharpening.
def sharp(img, sharp_point, sharp_low):
    # TODO customizable sliders for kernel parameters
    # TODO try darkening image
    s_kernel = np.array([[0, sharp_low, 0], [sharp_low, sharp_point, sharp_low], [0, sharp_low, 0]])

    sharpened = filter2D(img, -1, s_kernel)
    # plt.subplot(121)
    # plt.imshow(img2)
    # plt.title('Original')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(122)
    # plt.imshow(sharpened)
    # plt.title('sharp')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    return sharpened

def auto_canny(image, sigma=0):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    
    return edged

# function will call the blur and sharpen on every file in directory, and write output file
def removeScreentones(image, blur_amount, sh_point=5.56, sh_low=-1.14):

    # calculate sh params, warning if they are unproportionate
    sh_point = float(sh_point)
    sh_low = float(sh_low)
    print(sh_point, sh_low)
    sharps = (4 * sh_low) + sh_point - 1 # weight is initially just 1
#     print(sharpsto the sharpness of the screentones causing them
# to be confused with edge lines)
    bs_amount = 0
    if(blur_amount==1):
        bs_amount=3
    if(blur_amount==2):
        bs_amount=5
    if(blur_amount==3):
        bs_amount=7

    blurred = blur(image, bs_amount)
    ret = sharp(blurred, sh_point, sh_low)
    
    return ret

def _brightness(image, min=0.5, max=2.0):
    '''
    Randomly changes the brightness of the input image.
 
    Protected against overflow.
    '''
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    random_br = np.random.uniform(min,max)

    #To protect against overflow: Calculate a mask for all pixels
    #where adjustment of the brightness would exceed the maximum
    #brightness value and set the value to the maximum at those pixels.
    mask = hsv[:,:,2] * random_br > 255
    v_channel = np.where(mask, 255, hsv[:,:,2] * random_br)
    hsv[:,:,2] = v_channel

def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).
    
    # Arguments
        radius: radius of ball shape.
             
    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def get_unfilled_point(image):
    """Get points belong to unfilled(value==255) area.
    # Arguments
        image: an image.
    # Returns
        an array of points.
    """
    y, x = np.where(image == 255)

    return np.stack((x.astype(int), y.astype(int)), axis=-1)


def exclude_area(image, radius):
    """Perform erosion on image to exclude points near the boundary.
    We want to pick part using floodfill from the seed point after dilation. 
    When the seed point is near boundary, it might not stay in the fill, and would
    not be a valid point for next floodfill operation. So we ignore these points with erosion.
    # Arguments
        image: an image.
        radius: radius of ball shape.
    # Returns
        an image after dilation.
    """
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)


def trapped_ball_fill_single(image, seed_point, radius):
    """Perform a single trapped ball fill operation.
    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
        radius: radius of ball shape.
    # Returns
        an image after filling.
    """
    ball = get_ball_structuring_element(radius)

    pass1 = np.full(image.shape, 255, np.uint8)
    pass2 = np.full(image.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(image)

    # Floodfill the image
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    # Perform dilation on image. The fill areas between gaps became disconnected.
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)
    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

    # Floodfill with seed point again to select one fill area.
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    # Perform erosion on the fill result leaking-proof fill.
    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)

    return pass2


def trapped_ball_fill_multi(image, radius, method='mean', max_iter=1000):
    """Perform multi trapped ball fill operations until all valid areas are filled.
    # Arguments
        image: an image. The image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        radius: radius of ball shape.
        method: method for filtering the fills. 
               'max' is usually with large radius for select large area such as background.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    print('trapped-ball ' + str(radius))

    unfill_area = image
    filled_area, filled_area_size, result = [], [], []

    for _ in range(max_iter):
        points = get_unfilled_point(exclude_area(unfill_area, radius))

        if not len(points) > 0:
            break

        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size)

    if method == 'max':
        area_size_filter = np.max(filled_area_size)
    elif method == 'median':
        area_size_filter = np.median(filled_area_size)
    elif method == 'mean':
        area_size_filter = np.mean(filled_area_size)
    else:
        area_size_filter = 0

    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    for i in result_idx:
        result.append(filled_area[i])

    return result


def flood_fill_single(im, seed_point):
    """Perform a single flood fill operation.
    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
    # Returns
        an image after filling.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def flood_fill_multi(image, max_iter=20000):
    """Perform multi flood fill operations until all valid areas are filled.
    This operation will fill all rest areas, which may result large amount of fills.
    # Arguments
        image: an image. the image should contain white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    print('floodfill')

    unfill_area = image
    filled_area = []

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)

        if not len(points) > 0:
            break

        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))

    return filled_area


def mark_fill(image, fills):
    """Mark filled areas with 0.
    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result


def build_fill_map(image, fills):
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.
    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an array.
    """
    result = np.zeros(image.shape[:2], np.int)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap):
    """Mark filled areas with colors. It is useful for visualization.
    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def get_bounding_rect(points):
    """Get a bounding rect of points.
    # Arguments
        points: array of points.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2


def get_border_bounding_rect(h, w, p1, p2, r):
    """Get a valid bounding rect in the image with border of specific size.
    # Arguments
        h: image max height.
        w: image max width.
        p1: start point of rect.
        p2: end point of rect.
        r: border radius.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if 0 < x1 - r else 0
    y1 = y1 - r if 0 < y1 - r else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return x1, y1, x2, y2


def get_border_point(points, rect, max_height, max_width):
    """Get border points of a fill area
    # Arguments
        points: points of fill .
        rect: bounding rect of fill.
        max_height: image max height.
        max_width: image max width.
    # Returns
        points , convex shape of points
    """
    # Get a local bounding rect.
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect.
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    # Move points to the rect.
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    # Structuring element in cross shape is used instead of box to get 4-connected border.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill
    border_pixel_points = np.where(border_pixel_mask == 255)

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap, max_iter=10):
    """Merge fill areas.
    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for i in range(max_iter):
        print('merge ' + str(i + 1))

        result[np.where(fillmap == 0)] = 0

        fill_id = np.unique(result.flatten())
        fills = []

        for j in fill_id:
            point = np.where(result == j)

            fills.append({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point)
            })

        for j, f in enumerate(fills):
            # ignore lines
            if f['id'] == 0:
                continue

            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:
                    new_id = 0
            else:
                # region id may be set to region with largest contact
                new_id = ids[0]

            # a point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            #
            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:
                result[f['point']] = new_id

            if f['area'] < 50:
                result[f['point']] = new_id

        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result