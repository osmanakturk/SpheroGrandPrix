import cv2 as cv
import numpy as np


def trackbar(val):
    pass

cap = cv.VideoCapture(1, cv.CAP_DSHOW)


cv.namedWindow("Original Frame", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Mask - PURPLE", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Filtered - PURPLE", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Trackbars - PURPLE", cv.WINDOW_NORMAL)


cv.createTrackbar("H Min", "Trackbars - PURPLE", 130, 179, trackbar)
cv.createTrackbar("H Max", "Trackbars - PURPLE", 160, 179, trackbar)
cv.createTrackbar("S Min", "Trackbars - PURPLE", 50, 255, trackbar)
cv.createTrackbar("S Max", "Trackbars - PURPLE", 255, 255, trackbar)
cv.createTrackbar("V Min", "Trackbars - PURPLE", 50, 255, trackbar)
cv.createTrackbar("V Max", "Trackbars - PURPLE", 255, 255, trackbar)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hmin = cv.getTrackbarPos("H Min", "Trackbars - PURPLE")
    hmax = cv.getTrackbarPos("H Max", "Trackbars - PURPLE")
    smin = cv.getTrackbarPos("S Min", "Trackbars - PURPLE")
    smax = cv.getTrackbarPos("S Max", "Trackbars - PURPLE")
    vmin = cv.getTrackbarPos("V Min", "Trackbars - PURPLE")
    vmax = cv.getTrackbarPos("V Max", "Trackbars - PURPLE")

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask - PURPLE", mask)
    cv.imshow("Filtered - PURPLE", result)

    key = cv.waitKey(1) & 0xFF
    if key == ord('s'):
        print("*"*20)
        print(" Current PURPLE range:")
        print("\"Purple\"  : {\"Lower\" :("+str(hmin)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(hmax)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("*"*20)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
