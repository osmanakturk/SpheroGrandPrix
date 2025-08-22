import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time, glob, os
from ultralytics import YOLO, SAM, FastSAM


#cap = cv.VideoCapture(1, cv.CAP_DSHOW)

cap = cv.VideoCapture("./sphero1.mp4")


model = YOLO("./models/yolo11n-seg.pt")


fgbg_KNN = cv.createBackgroundSubtractorKNN(history=500, detectShadows=True)

canvas = None

prew_pt = (None, None)



if not cap.isOpened():
    exit()


while True:

    ret, frame = cap.read()
    

    if not ret:
        break

    frame = frame[:, 270:550]
    #frame = frame[:, 200:440]


 
    

    fgmask_KNN = fgbg_KNN.apply(frame.copy())

    _, fgmask_thr  = cv.threshold(fgmask_KNN, 127, 255, cv.THRESH_BINARY)
    _, bgmask_thr  = cv.threshold(fgmask_KNN, 127, 255, cv.THRESH_BINARY_INV)


    bg_model_KNN = fgbg_KNN.getBackgroundImage()

    bgmask_KNN = cv.bitwise_not(fgmask_KNN)

    ball_thr = cv.bitwise_and(frame.copy(), frame.copy(), mask=fgmask_thr)
    
    blank = np.full_like(frame, 255, np.uint8)

    ball_blank = cv.bitwise_not(blank, blank, mask=fgmask_thr)
 
    ball_full = cv.add(ball_thr, ball_blank)

    cv.imshow('ball_full', ball_full)


    test_total = cv.bitwise_or(frame.copy(), ball_thr.copy())




    
    cv.imshow('ball_thr', ball_thr)
  
    cv.imshow("bg_model_KNN", bg_model_KNN)


    cv.imshow("fgmask_thr", fgmask_thr)
    cv.imshow("bgmask_thr", bgmask_thr)


    cv.imshow("fgmask_KNN", fgmask_KNN)
    cv.imshow("bgmask_KNN", bgmask_KNN)




    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Alternative segmentation: tracker="botsort.yaml", tracker="bytetrack.yaml"
    results = model.track(ball_full, stream=False, persist=False, verbose=False, classes=[9, 32], half=False)
    result = results[0]
    img = result.plot(conf=False, labels=False)
   
   

    bboxs = result.boxes

    
    

    for box in bboxs:
       
        cx, cy, w, h = box.xywh[0].int().tolist()
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        roi = frame[y1:y2, x1:x2]
        cv.imshow("roi", roi)
        if prew_pt[0] is None and prew_pt[1] is None:
            prew_pt = (cx, cy)
        else:
            
            cv.line(canvas, prew_pt, (cx, cy), color=(0, 0, 255), thickness=3, lineType=cv.LINE_AA)
            prew_pt = (cx, cy)

    




    frame = cv.add(frame, canvas)

    cv.imshow("Camera", frame)
    cv.imshow("YOLO", img)
    cv.imshow("canvas", canvas)

    

    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv.destroyAllWindows()








