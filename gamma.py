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
from ex1_utils import *
import cv2 as cv


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    # reading the image based on the desired representation
    try:
        if rep == LOAD_GRAY_SCALE:
            img = cv.imread(img_path, 2)
        else:
            img = cv.imread(img_path, 1)
    except:
        print("The image path was not valid")

    # normalizing the image
    img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    # creating a copy that we will work on
    tempImg = img.copy()
    # creating new window
    cv.namedWindow('gamma correction')
    # creating a trackbar
    cv.createTrackbar('curr gamma', 'gamma correction', 100, 200, emptyFunc)
    # close the window to finish the program
    while cv.getWindowProperty('gamma correction', cv.WND_PROP_VISIBLE) > 0:
        # resizing the image so it will fit in the screen
        img_resize = cv.resize(img, (960, 540))
        cv.imshow('gamma correction', img_resize)
        # the gamma of the slider
        curr_bar_gamma = cv.getTrackbarPos('curr gamma', 'gamma correction')
        # the real gamma is the slider gamma divided by 100
        real_gamma = curr_bar_gamma / 100
        # the image based on the real gamma as a formula we have been taught
        img = np.power(tempImg, real_gamma)
        if cv.waitKey(100) > 0:
            break
    cv.destroyAllWindows()

# empty function because the trackbar must get a function,
# while im doing all of the gamma adjustment in the gammaDisplay function
def emptyFunc(x):
    pass


def main():
    gammaDisplay('testImg2.jpg', LOAD_RGB)


if __name__ == '__main__':
    main()
