import cv2 as cv
import numpy as np


def convertToHSV(arr):
    BGR = np.uint8([[arr]])
    HSV = cv.cvtColor(BGR, cv.COLOR_BGR2HSV)
    return HSV

x = [70,86,155]
y = [134,122,198]
z = [84,106,147]

print(convertToHSV(x))
print(convertToHSV(y))
print(convertToHSV(z))