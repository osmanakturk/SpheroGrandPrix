import cv2 as cv
import numpy as np



def trackbar(val): 
    pass

cap = cv.VideoCapture(1, cv.CAP_DSHOW)





cv.namedWindow("Original Frame", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Mask - YELLOW", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Filtered - YELLOW", cv.WINDOW_KEEPRATIO)
cv.namedWindow("Trackbars - YELLOW", cv.WINDOW_NORMAL)




cv.createTrackbar("H Min", "Trackbars - YELLOW", 20, 179, trackbar)
cv.createTrackbar("H Max", "Trackbars - YELLOW", 30, 179, trackbar)
cv.createTrackbar("S Min", "Trackbars - YELLOW", 100, 255, trackbar)
cv.createTrackbar("S Max", "Trackbars - YELLOW", 255, 255, trackbar)
cv.createTrackbar("V Min", "Trackbars - YELLOW", 100, 255, trackbar)
cv.createTrackbar("V Max", "Trackbars - YELLOW", 255, 255, trackbar)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break


    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    hmin = cv.getTrackbarPos("H Min", "Trackbars - YELLOW")
    hmax = cv.getTrackbarPos("H Max", "Trackbars - YELLOW")
    smin = cv.getTrackbarPos("S Min", "Trackbars - YELLOW")
    smax = cv.getTrackbarPos("S Max", "Trackbars - YELLOW")
    vmin = cv.getTrackbarPos("V Min", "Trackbars - YELLOW")
    vmax = cv.getTrackbarPos("V Max", "Trackbars - YELLOW")

    lower = np.array([hmin, smin, vmin], dtype=np.uint8)
    upper = np.array([hmax, smax, vmax], dtype=np.uint8)

    mask = cv.inRange(hsv, lower, upper)
    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("Original Frame", frame)
    cv.imshow("Mask - YELLOW", mask)
    cv.imshow("Filtered - YELLOW", result)

    key = cv.waitKey(1) & 0xFF

    if key == ord('c'):
        print("*"*20)
        print("Current YELLOW range:")
        print("\"Yellow\"  : {\"Lower\" :("+str(hmin)+", "+str(smin)+", "+str(vmin)+"), \"Upper\" : ("+str(hmax)+ ", "+str(smax)+", "+str(vmax)+")}")
        print("*"*20)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()
