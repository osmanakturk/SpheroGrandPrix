import cv2 as cv
import numpy as np
import time, glob, os, math, logging
from backend.constants import HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_HSV, COLORS_BGR



#cap = cv.VideoCapture(1, cv.CAP_DSHOW)

cap = cv.VideoCapture("./tests/sphero1.mp4")





background_subtractor_KNN = cv.createBackgroundSubtractorKNN(history=500, detectShadows=True)

canvas = None

prew_pt = (None, None)



if not cap.isOpened():
    exit()


while True:

    ret, frame = cap.read()
    

    if not ret:
        break

    # TODO SET FRAME ZONE
    frame = frame[:, 270:550]
    orginal = frame.copy()
    #frame = frame[:, 200:440]
    frame = cv.medianBlur(frame, 15)
    cv.imshow("frame orginal", frame)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    white_blank = np.full_like(frame, 255, np.uint8)
    black_blank = np.full_like(frame, 0, np.uint8)
    

    if canvas is None:
        canvas = black_blank


    fgmask_KNN = background_subtractor_KNN.apply(frame.copy())
    bgmask_KNN = cv.bitwise_not(fgmask_KNN)
    cv.imshow("fgmask_KNN", fgmask_KNN)
    #cv.imshow("bgmask_KNN", bgmask_KNN)

    _, fgmask_thr  = cv.threshold(fgmask_KNN, 127, 255, cv.THRESH_BINARY)
    _, bgmask_thr  = cv.threshold(fgmask_KNN, 127, 255, cv.THRESH_BINARY_INV)
    cv.imshow("fgmask_thr", fgmask_thr)
    #cv.imshow("bgmask_thr", bgmask_thr)

    bg_model_KNN = background_subtractor_KNN.getBackgroundImage()
    #cv.imshow("bg_model_KNN", bg_model_KNN)

    

    ball_thr = cv.bitwise_and(frame.copy(), frame.copy(), mask=fgmask_thr)
    cv.imshow('ball_thr', ball_thr)


    
    ball_blank = cv.bitwise_not(white_blank.copy(), white_blank.copy(), mask=fgmask_thr)
    ball_full = cv.add(ball_thr, ball_blank)
    cv.imshow('ball_full', ball_full)
    



    mask_yellow = cv.inRange(hsv, HSV_RANGES_WIDE["Yellow"]["Lower"], HSV_RANGES_WIDE["Yellow"]["Upper"])
    mask_yellow = cv.bitwise_and(frame, frame, mask=mask_yellow)
    yellow_ball = cv.min(mask_yellow, ball_thr)
    cv.imshow("yellow_ball", yellow_ball)
    yellow_ball_gray = cv.cvtColor(mask_yellow, cv.COLOR_BGR2GRAY)
    cv.imshow("yellow_ball_gray", yellow_ball_gray)
    _, thresh_yellow = cv.threshold(yellow_ball_gray, 150, 255, cv.THRESH_BINARY)
    cv.imshow("thresh_yellow", thresh_yellow)
    contours, hierarchy = cv.findContours(thresh_yellow, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        area = cv.contourArea(cont)
        if area > 300: #TODO SET AREA VALUE
            x, y, w, h = cv.boundingRect(cont)
            center = ((x + w//2), (y + h//2))
            radius = (w + h)//4
            #print(f"1 area: {area}, x: {x}, y: {y}, w: {w}, h: {h}, radius: {radius}, w*h: {w*h}, circle: {int(math.pi*radius*radius)}")
            (x1, y1), radius1 = cv.minEnclosingCircle(cont)
            x1, y1 = int(x1), int(y1)
            center1 = (x1, y1)
            radius1 = int(radius1)
            #print(f"2 area: {area}, x: {x}, y: {y}, w: {w}, h: {h}, radius: {radius}, w*h: {w*h}, circle: {int(math.pi*radius*radius)}")
            print(f"circle1: {center, radius}, area: {int(math.pi*radius*radius)}")            
            print(f"circle2: {center1, radius1}, area: {int(math.pi*radius1*radius1)}")
            cv.circle(thresh_yellow, center1, radius1, 255, -1, cv.LINE_AA)
            cv.circle(orginal, center1, radius1, COLORS_BGR["Yellow"], -1, cv.LINE_AA)
            cv.circle(orginal, center1, radius1+3, COLORS_BGR["Cyan"], 3, cv.LINE_AA)
            cv.putText(orginal, "User", (x1, y1-2*radius1), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR["Yellow"], 2, cv.LINE_AA)
            #frame = cv.addWeighted(orginal, 0.5, frame, 0.5, 0)


            if prew_pt[0] is None and prew_pt[1] is None:
                prew_pt = center1
            else:
                cv.line(canvas, prew_pt, center1, color=(0, 0, 255), thickness=3, lineType=cv.LINE_AA)
                prew_pt = center1


    cv.imshow("yellow_ball_gray_circled", thresh_yellow)


    mask_blue = cv.inRange(hsv, HSV_RANGES_WIDE["Blue"]["Lower"], HSV_RANGES_WIDE["Blue"]["Upper"])
    mask_blue = cv.bitwise_and(frame, frame, mask=mask_blue)
    #cv.imshow("mask_blue", mask_blue)


    mask_green = cv.inRange(hsv, HSV_RANGES_WIDE["Green"]["Lower"], HSV_RANGES_WIDE["Green"]["Upper"])
    mask_green = cv.bitwise_and(frame, frame, mask=mask_green)
    #cv.imshow("mask_green", mask_green)


    mask_purple = cv.inRange(hsv, HSV_RANGES_WIDE["Purple"]["Lower"], HSV_RANGES_WIDE["Purple"]["Upper"])
    mask_purple = cv.bitwise_and(frame, frame, mask=mask_purple)
    #cv.imshow("mask_purple", mask_purple)


    mask_red1 = cv.inRange(hsv, HSV_RANGES_WIDE["Red1"]["Lower"], HSV_RANGES_WIDE["Red1"]["Upper"])
    mask_red2 = cv.inRange(hsv, HSV_RANGES_WIDE["Red2"]["Lower"], HSV_RANGES_WIDE["Red2"]["Upper"])
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_red = cv.bitwise_and(frame, frame, mask=mask_red)
    #cv.imshow("mask_red", mask_red)   







    camera = cv.add(orginal, canvas)

    #cv.imshow("YOLO", img)

    cv.imshow("Camera", camera)
    
    cv.imshow("Canvas", canvas)

    

    time.sleep(0.1)

    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv.destroyAllWindows()








