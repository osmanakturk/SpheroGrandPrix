import cv2 as cv
import numpy as np


def trackbar(val): 
    pass


cap = cv.VideoCapture(2, cv.CAP_DSHOW)


cv.namedWindow("Original Frame", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Mask - RED", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Filtered - RED", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Trackbars - RED", cv.WINDOW_NORMAL)



cv.createTrackbar("H1 Min", "Trackbars - RED", 0, 179, trackbar)
cv.createTrackbar("H1 Max", "Trackbars - RED", 10, 179, trackbar)
cv.createTrackbar("H2 Min", "Trackbars - RED", 160, 179, trackbar)
cv.createTrackbar("H2 Max", "Trackbars - RED", 179, 179, trackbar)
cv.createTrackbar("S Min", "Trackbars - RED", 100, 255, trackbar)
cv.createTrackbar("S Max", "Trackbars - RED", 255, 255, trackbar)
cv.createTrackbar("V Min", "Trackbars - RED", 100, 255, trackbar)
cv.createTrackbar("V Max", "Trackbars - RED", 255, 255, trackbar)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret: 
        break



    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

 
    h1min = cv.getTrackbarPos("H1 Min", "Trackbars - RED")
    h1max = cv.getTrackbarPos("H1 Max", "Trackbars - RED")
    h2min = cv.getTrackbarPos("H2 Min", "Trackbars - RED")
    h2max = cv.getTrackbarPos("H2 Max", "Trackbars - RED")
    smin  = cv.getTrackbarPos("S Min", "Trackbars - RED")
    smax  = cv.getTrackbarPos("S Max", "Trackbars - RED")
    vmin  = cv.getTrackbarPos("V Min", "Trackbars - RED")
    vmax  = cv.getTrackbarPos("V Max", "Trackbars - RED")

    lower1 = np.array([h1min, smin, vmin], dtype=np.uint8)
    upper1 = np.array([h1max, smax, vmax], dtype=np.uint8)
    lower2 = np.array([h2min, smin, vmin], dtype=np.uint8)
    upper2 = np.array([h2max, smax, vmax], dtype=np.uint8)

 
    mask1 = cv.inRange(hsv, lower1, upper1)
    mask2 = cv.inRange(hsv, lower2, upper2)
    mask  = cv.bitwise_or(mask1, mask2)

    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask - RED", mask)
    cv.imshow("Filtered - RED", result)

    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        print("*"*20)
        print("Current RED ranges:")
        print("\"Red1\"  : {\"Lower\" :("+str(h1min)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(h1max)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("\"Red2\"  : {\"Lower\" :("+str(h2min)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(h2max)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("*"*20)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()

