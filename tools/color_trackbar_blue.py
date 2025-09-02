import cv2 as cv
import numpy as np


def trackbar(val):
    pass

cap = cv.VideoCapture(1, cv.CAP_DSHOW)



cv.namedWindow("Original Frame", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Mask - BLUE", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Filtered - BLUE", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Trackbars - BLUE", cv.WINDOW_NORMAL)


cv.createTrackbar("H Min", "Trackbars - BLUE", 90, 179, trackbar)
cv.createTrackbar("H Max", "Trackbars - BLUE", 130, 179, trackbar)
cv.createTrackbar("S Min", "Trackbars - BLUE", 70, 255, trackbar)
cv.createTrackbar("S Max", "Trackbars - BLUE", 255, 255, trackbar)
cv.createTrackbar("V Min", "Trackbars - BLUE", 70, 255, trackbar)
cv.createTrackbar("V Max", "Trackbars - BLUE", 255, 255, trackbar)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hmin = cv.getTrackbarPos("H Min", "Trackbars - BLUE")
    hmax = cv.getTrackbarPos("H Max", "Trackbars - BLUE")
    smin = cv.getTrackbarPos("S Min", "Trackbars - BLUE")
    smax = cv.getTrackbarPos("S Max", "Trackbars - BLUE")
    vmin = cv.getTrackbarPos("V Min", "Trackbars - BLUE")
    vmax = cv.getTrackbarPos("V Max", "Trackbars - BLUE")

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask - BLUE", mask)
    cv.imshow("Filtered - BLUE", result)

    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        print("*"*20)
        print(" Current BLUE range:")
        print("\"Blue\"  : {\"Lower\" :("+str(hmin)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(hmax)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("*"*20)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
