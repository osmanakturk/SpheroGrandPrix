
import cv2 as cv
import numpy as np
import time





def trackbar_callback(val):
    #print(val)
    pass



#cap = cv.VideoCapture("./sphero/sphero1.mp4")
cap = cv.VideoCapture(1, cv.CAP_DSHOW)

cap.set(cv.CAP_PROP_SETTINGS, 1)


cv.namedWindow("Perspective", cv.WINDOW_NORMAL)
cv.namedWindow("Trackbar", cv.WINDOW_NORMAL)
cv.namedWindow("Camera", cv.WINDOW_KEEPRATIO)

FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))



cv.createTrackbar("TL_x", "Trackbar", 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("TL_y", "Trackbar", 0, FRAME_HEIGHT, trackbar_callback)

cv.createTrackbar("TR_x", "Trackbar", 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("TR_y", "Trackbar", 0, FRAME_HEIGHT, trackbar_callback)


cv.createTrackbar("BL_x", "Trackbar", 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("BL_y", "Trackbar", 0, FRAME_HEIGHT, trackbar_callback)

cv.createTrackbar("BR_x", "Trackbar", 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("BR_y", "Trackbar", 0, FRAME_HEIGHT, trackbar_callback)


cv.createTrackbar("F_y", "Perspective", 0, FRAME_HEIGHT, trackbar_callback)

while cap.isOpened():
    

    ret, frame = cap.read()


    if not ret:
        break


 
    tl_x = int(cv.getTrackbarPos("TL_x", "Trackbar"))
    tl_y = int(cv.getTrackbarPos("TL_y", "Trackbar"))

    tr_x = int(cv.getTrackbarPos("TR_x", "Trackbar"))
    tr_y = int(cv.getTrackbarPos("TR_y", "Trackbar"))

    bl_x = int(cv.getTrackbarPos("BL_x", "Trackbar"))
    bl_y = int(cv.getTrackbarPos("BL_y", "Trackbar"))

    br_x = int(cv.getTrackbarPos("BR_x", "Trackbar"))
    br_y = int(cv.getTrackbarPos("BR_y", "Trackbar"))


    finishline = int(cv.getTrackbarPos("Finishline", "Perspective"))
    

    
    canvas = np.full_like(frame, 0, np.uint8)
    canvas[min(tl_y, tr_y):max(bl_y, br_y)+1, min(tl_x, bl_x):max(tr_x, br_x)+1] = frame[min(tl_y, tr_y):max(bl_y, br_y)+1, min(tl_x, bl_x):max(tr_x, br_x)+1]

    
    cv.line(frame, (tl_x, tl_y), (tr_x, tr_y), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(frame, (tl_x, tl_y), (bl_x, bl_y), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(frame, (tr_x, tr_y), (br_x, br_y), (0, 0, 255), 2, cv.LINE_AA)
    cv.line(frame, (bl_x, bl_y), (br_x, br_y), (0, 0, 255), 2, cv.LINE_AA)




    cv.putText(frame, f"TL:{tl_x, tl_y}", (tl_x, tl_y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, f"TR:{tr_x, tr_y}", (tr_x, tr_y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, f"BL:{bl_x, bl_y}", (bl_x, bl_y+15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(frame, f"BR:{br_x, br_y}", (br_x, br_y+15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
    

    if any([tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, finishline]):

        x_max = max(abs(tl_x - tr_x), abs(bl_x - br_x)) | 0
        y_max = max(abs(tl_y - bl_y), abs(tr_y - br_y)) | 0

        width = x_max
        height = y_max



        pts_src = np.array([[tl_x, tl_y], 
                            [tr_x, tr_y], 
                            [br_x, br_y],
                            [bl_x, bl_y]], dtype=np.float32)
        
        pts_dst = np.array([[0, 0], 
                            [width, 0], 
                            [width, height], 
                            [0, height]], dtype=np.float32)

        matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
        perspective = cv.warpPerspective(frame, matrix, (width, height))
        cv.line(perspective, (0, finishline), (width, finishline), (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(perspective, f"Finishline y:({finishline})", (0, finishline-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.imshow("Perspective", perspective)

    

    cv.imshow("Camera", frame)
    cv.imshow("Canvas", canvas)
    


    key = cv.waitKey(1)


    if key & 0xFF == 27:
        break
    elif key == ord("c"):

        try:
            print("*"*20)
            print(f"Top-Left: {tl_x, tl_y}")
            print(f"Top_Right: {tr_x, tr_y}") 
            print(f"Bottom-Left: {bl_x, bl_y}") 
            print(f"Bottom-Right: {br_x, br_y}")
            print(f"Finishline y: ({finishline})")
            print(f"Area: [{min(tl_y, tr_y)}:{max(bl_y, br_y)}, {min(tl_x, bl_x)}:{max(tr_x, br_x)}]")
            print(f"x_max: {x_max}, y_max: {y_max}")
            print(f"width: {width} height: {height}")
            print("*"*20)
        except Exception as e:
            print(e)




    
cap.release()
cv.destroyAllWindows()