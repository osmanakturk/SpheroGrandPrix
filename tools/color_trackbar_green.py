import cv2 as cv
import numpy as np


def trackbar(val):
    pass


cap = cv.VideoCapture(1, cv.CAP_DSHOW)


cv.namedWindow("Original Frame", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Mask - GREEN", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Filtered - GREEN", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Trackbars - GREEN", cv.WINDOW_NORMAL)


cv.createTrackbar("H Min", "Trackbars - GREEN", 40, 179, trackbar)
cv.createTrackbar("H Max", "Trackbars - GREEN", 85, 179, trackbar)
cv.createTrackbar("S Min", "Trackbars - GREEN", 70, 255, trackbar)
cv.createTrackbar("S Max", "Trackbars - GREEN", 255, 255, trackbar)
cv.createTrackbar("V Min", "Trackbars - GREEN", 70, 255, trackbar)
cv.createTrackbar("V Max", "Trackbars - GREEN", 255, 255, trackbar)

while cap.isOpened():
    
    ret, frame = cap.read()

    if not ret: break


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hmin = cv.getTrackbarPos("H Min", "Trackbars - GREEN")
    hmax = cv.getTrackbarPos("H Max", "Trackbars - GREEN")
    smin = cv.getTrackbarPos("S Min", "Trackbars - GREEN")
    smax = cv.getTrackbarPos("S Max", "Trackbars - GREEN")
    vmin = cv.getTrackbarPos("V Min", "Trackbars - GREEN")
    vmax = cv.getTrackbarPos("V Max", "Trackbars - GREEN")

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask - GREEN", mask)
    cv.imshow("Filtered - GREEN", result)

    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        print("*"*20)
        print("Current GREEN range:")
        print("\"Green\"  : {\"Lower\" :("+str(hmin)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(hmax)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("*"*20)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
