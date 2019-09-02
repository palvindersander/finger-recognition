import cv2 as cv
import numpy as np


def convertToHSV(arr):
    BGR = np.uint8([[arr]])
    HSV = cv.cvtColor(BGR, cv.COLOR_BGR2HSV)
    return HSV


def loadImage():
    img = cv.imread('sample2.jpg')
    return img


def findHand(img):
    while(True):
        kernel = np.ones((3,3),np.uint8)
        LOWER = np.array([143,59,50], dtype=np.uint8)
        UPPER= np.array([173,108,255], dtype=np.uint8)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, LOWER, UPPER)
        mask = cv.dilate(mask,kernel,iterations=50)
        mask = cv.GaussianBlur(mask,(5,5),100)
        res = cv.bitwise_and(img, img, mask=mask)
        
        imgS = cv.resize(img, (500, 500))
        maskS = cv.resize(mask, (500, 500))
        resS = cv.resize(res, (500, 500))
        cv.imshow('img', imgS)
        cv.imshow('mask', maskS)
        cv.imshow('res', resS)
        k = cv.waitKey(33)
        if k == 27:
            break


findHand(loadImage())
cv.destroyAllWindows()
