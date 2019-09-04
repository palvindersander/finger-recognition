import cv2 as cv
import numpy as np
from findHSV import main

def rescale_frame(frame, wpercent=130, hpercent=130):
    width = int(frame.shape[1] * wpercent / 100)
    height = int(frame.shape[0] * hpercent / 100)
    return cv,resize(frame, (width, height), interpolation=cv.INTER_AREA)

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

def frameAnalysis(img):
    fingerNumber = 0
    img = cv.medianBlur(img, 5)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    maskLower = np.array([0, 37, 46])
    maskHigher = np.array([21, 185, 239])
    mask = cv.inRange(hsv, maskLower, maskHigher)
    kernel = np.ones((2, 2), np.uint8)
    mask = cv.dilate(mask, kernel, iterations=10)
    mask = cv.GaussianBlur(mask, (5, 5), 0)
    res = cv.bitwise_and(img, img, mask=mask)
    ret, thresh = cv.threshold(mask, 127, 255, 0)
    contours, hierarchy = cv.findContours(
        thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    handContour = getMaxContours(contours)
    cv.drawContours(img, handContour, -1, (0, 255, 0), 20)
    hull = cv.convexHull(handContour, returnPoints=False)
    convexHullPoints = cv.convexHull(
        handContour, returnPoints=True, clockwise=True)
    defects = cv.convexityDefects(handContour, hull)
    allDefects = []
    fingerDefects = []
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(handContour[s][0])
        end = tuple(handContour[e][0])
        far = tuple(handContour[f][0])
        angle = calculateAngle(far, start, end)
        allDefects.append(far)
        cv.line(img, start, end, [0, 0, 255], 20)
        if d > 100000 and angle <= (math.pi/9)*8:
            fingerDefects.append(far)
            cv.circle(img, far, 25, [255, 0, 0], -1)
            fingerNumber = fingerNumber + 1
    if fingerNumber > 1:
        fingerNumber = fingerNumber + 1
    x, y, w, h = cv.boundingRect(handContour)
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 10)
    font = cv.FONT_ITALIC
    cv.putText(img, str(fingerNumber), (0, 300), font,
                10, (0, 255, 255), 20, cv.LINE_AA)
    moments = cv.moments(thresh)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    cv.circle(img, (cX, cY), 25, (255, 255, 255), -1)
    if fingerNumber == 1:
        dist = 0
        p = []
        for point in convexHullPoints:
            point = point[0]
            distance = math.sqrt(
                ((point[1] - cY)**2) + ((cX - point[0])**2))
            if distance > dist:
                dist = distance
                p = point
        cv.circle(img, (p[0], p[1]), 25, (255, 255, 255), -1)
    return img

def main():
    capture = cv.VideoCapture(0)

    while capture.isOpened():
        pressed_key = cv.waitKey(1)
        _, frame = capture.read()

        if pressed_key & 0xFF == ord('z'):
            main()

        frame = frameAnalysis(frame)

        cv.imshow("Live Feed", rescale_frame(frame))

        if pressed_key == 27:
            break

    cv.destroyAllWindows()
    capture.release()

if __name__ == '__main__':
    main()