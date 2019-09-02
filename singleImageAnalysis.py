import cv2 as cv
import numpy as np
import math

def convertToHSV(arr):
    BGR = np.uint8([[arr]])
    HSV = cv.cvtColor(BGR, cv.COLOR_BGR2HSV)
    return HSV


def loadImage():
    img = cv.imread('images/sample.jpg')
    return img

def getMaxContours(contours):
    maxIndex = 0
    maxArea = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        if area > maxArea:
            maxArea = area
            maxIndex = i
    return contours[maxIndex]

def calculateAngle(far, start, end):
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def findHand():
    while(1):
        fingerNumber = 0
        img = loadImage()
        k = cv.waitKey(1)
        #img = cv.medianBlur(img,5)
        img = cv.GaussianBlur(img,(5,5),0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        maskLower = np.array([0,40,189])
        maskHigher = np.array([8,108,255])
        mask = cv.inRange(hsv, maskLower, maskHigher)

        #for uneven skintone
        maskUnevenLower = np.array([0,65,70])
        maskUnevenHigher = np.array([10,200,200])
        maskUneven = cv.inRange(hsv,maskUnevenLower,maskUnevenHigher)

        maskUnevenLower = np.array([0,65,70])
        maskUnevenHigher = np.array([10,200,200])
        maskUneven = cv.inRange(hsv,maskUnevenLower,maskUnevenHigher)

        mask = mask + maskUneven + cv.inRange(hsv,np.array([160,75,75]),np.array([255,255,255]))

        kernel = np.ones((2,2),np.uint8)
        #mask = cv.dilate(mask,kernel,iterations=1)  
        res = cv.bitwise_and(img, img, mask=mask)

        resGrey = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(resGrey, 127, 255, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        handContour = getMaxContours(contours)
        cv.drawContours(img, handContour, -1, (0,255,0), 20)
        hull = cv.convexHull(handContour, returnPoints=False)
        defects = cv.convexityDefects(handContour,hull)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(handContour[s][0])
            end = tuple(handContour[e][0])
            far = tuple(handContour[f][0])
            angle = calculateAngle(far, start, end)
            cv.line(img,start,end,[0,0,255],20)
            if d > 100000 and angle <=(math.pi/9)*8:
                cv.circle(img,far,25,[255,0,0],-1)
                fingerNumber = fingerNumber + 1
        x,y,w,h = cv.boundingRect(handContour)
        cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),10)
        font = cv.FONT_ITALIC
        fingerNumber = fingerNumber + 1
        cv.putText(img,str(fingerNumber),(0,300), font, 10,(0,255,255),20,cv.LINE_AA)

        
        imgSmall = cv.resize(img, (500, 500))
        maskSmall = cv.resize(mask, (500, 500))
        resSmall = cv.resize(res, (500, 500))
        cv.imshow('img', imgSmall)
        cv.imshow('mask', maskSmall)
        cv.imshow('res', resSmall)
        if k == 27:
            break
    cv.destroyAllWindows()


findHand()
