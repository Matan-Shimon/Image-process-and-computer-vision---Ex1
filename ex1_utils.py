"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 314669342


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # read the image
    try:
        image = cv.imread(filename)
    except:
        print("The image path was not valid")
    converted_image = 0
    if representation == LOAD_GRAY_SCALE: # if it'sa gray scale image
        # converting the image from defalut (BGR) to gray
        converted_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    elif representation == LOAD_RGB: # if it's an RGB image
        # converting the image from defalut (BGR) to RGB
        converted_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    else:
        return "Representation can be only 1 for gray scale  and 2 for RGB"
    # normalizing the converted image
    image_norm = cv.normalize(converted_image, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return image_norm


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    # reading and converting the image by the desired representation
    img = imReadAndConvert(filename, representation)
    # if it's a gray scale image we will show it by using cmap='gray', else, we will show by default
    if representation == 1:
        plt.imshow(img, cmap='gray')
        plt.title('Gray scale')
    elif representation == 2:
        plt.imshow(img)
        plt.title('RGB')
    else:
        return "Representation can be only 1 for gray scale  and 2 for RGB"
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    # the matrix that will help us to convert from RGB to YIQ
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    # using dot product action to get the desired output
    try:
        YIQ = np.dot(imgRGB, mat.transpose())
    except:
        print("The image was not valid")
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    # the matrix that will help us to convert from YIQ to RGB
    mat = np.array([[0.299, 0.587, 0.114],
                    [0.596, -0.275, -0.321],
                    [0.212, -0.523, 0.311]])
    # calculating the inverse matrix
    inverse_mat = np.linalg.inv(mat)
    # using dot product action to get the desired output
    try:
        RGB = np.dot(imgYIQ, inverse_mat.transpose())
    except:
        print("The image was not valid")
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    # if the image is RGB type then we need to convert the RGB to YIQ and work only on the Y part
    isRGB = False
    if len(imgOrig.shape) == 3:
        imgYIQ = transformRGB2YIQ(imgOrig)
        imgY = imgYIQ[:, :, 0]
        imgOrig = imgY
        isRGB = True
    # normalizing the image to 0-255
    imgOrig = cv.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    imgOrig = imgOrig.astype('uint8')
    # making a copy that we will work on
    imgEq = imgOrig.copy()
    # calculating the original image histogram
    histOrg = np.histogram(imgOrig.flatten(), bins=256)[0]
    # calculating the original image cumulative sum
    cum_sum = np.cumsum(histOrg)
    # calculating the max pixel for the look up table
    max_pix = cum_sum.max()
    # the look up table formula
    LUT = np.ceil((cum_sum / max_pix) * 255)
    # changing the original image based on the look up table
    for i in range(256):
        imgEq[imgOrig == i] = int(LUT[i])
    # flatting the updated image for the histogram calculation
    flatImg = imgEq.flatten().astype('uint8')
    # calculating the updated image histogram
    histEQ = np.histogram(flatImg, bins=256)[0]
    # normalizing back to 0-1
    imgEq = cv.normalize(imgEq, None, 0, 1, cv.NORM_MINMAX, dtype=cv.CV_32F)
    # if the image is RGB type
    if isRGB:
        imgYIQ[:, :, 0] = imgEq
        imgEq = transformYIQ2RGB(imgYIQ)
    return imgEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    # if the error will be less then EPS, we will stop making improvements
    if nQuant > 255 or nQuant < 1:
        return "number of colors (nQuant) must be 1-255"
    if nIter < 1:
        return "number of iterations must be positive!"
    EPS = 0.0001  # this will be efficient in case of large nIter
    # if the image is RGB type
    isRGB = False
    if len(imOrig.shape) == 3:
        # converting to YIQ and performing the improvements only on the Y channel
        imgYIQ = transformRGB2YIQ(imOrig)
        imgY = imgYIQ[:, :, 0]
        imOrig = imgY
        isRGB = True
    # flatting the image and presenting it in 0-255
    flatImg = (imOrig.flatten() * 255).astype('uint8')
    # calculating the histogram of the image
    hist = np.histogram(flatImg, bins=256)[0]
    # creating an array for the borders
    zBorders = []
    # split the border to even parts of pixels
    pix_sum = 0
    pix_index = 0
    zBorders.append(0)
    for i in range(1,nQuant + 1):
        while pix_sum < hist.sum()/nQuant and pix_index < 256:
            pix_sum += hist[pix_index]
            pix_index += 1
        if zBorders[-1] != pix_index-1:
            pix_index -= 1
        zBorders.append(pix_index)
        pix_sum = 0
    zBorders.pop()
    zBorders.append(255)
    # creating an empty lists for the desired output
    qImage_i = []
    error_i = []
    for k in range(nIter):
        updatedImg = np.zeros_like(imOrig)
        weightedAvgs = []
        for i in range(nQuant):
            # calculating the weighted average in each border
            intensity = hist[zBorders[i]:zBorders[i + 1]]
            intensity_indexes = range(zBorders[i], zBorders[i+1])
            intesity_sum = np.sum(intensity)
            # the formula we have been given for calculating the weighted average
            weightedAvg = (intensity * intensity_indexes).sum() / intesity_sum
            # in case we have a lot of borders the average in some images can be NaN because the
            # the borders change, to prevent that, we will replace it with 0, and by that we will
            # keep the same border it has before
            if np.isnan(weightedAvg):
                weightedAvg = 0
            weightedAvgs.append(weightedAvg)
        for i in range(nQuant):
            # updating the image
            updatedImg[imOrig*255 > zBorders[i]] = weightedAvgs[i]
        for i in range(nQuant-1):
            avg = (weightedAvgs[i] + weightedAvgs[i+1]) / 2
            zBorders[i+1] = int(avg)
        # adding the mse error to the errors output list
        error_i.append(((imOrig*255 - updatedImg) ** 2).mean(axis=None))
        # adding the updated image to the images output list
        qImage_i.append(updatedImg / 255)
        # if the error difference is so small, we will stop making improvements
        if len(error_i) > 1:
            if abs(error_i[-1] - error_i[-2]) < EPS:
                break
    # in case the error difference was so small
    if len(error_i) > 1:
        if abs(error_i[-1] - error_i[-2]) < EPS:
            last_error = error_i[-1]
            last_image = qImage_i[-1]
            # adding the last error and images
            for i in range(len(qImage_i), nIter):
                error_i.append(last_error)
                qImage_i.append(last_image)
    # if the image is RGB type
    if isRGB:
        for i in range(nIter):
            # converting back to RGB
            imgYIQ[:, :, 0] = qImage_i[i]
            RGB = transformYIQ2RGB(imgYIQ)
            qImage_i[i] = RGB
    return qImage_i, error_i
