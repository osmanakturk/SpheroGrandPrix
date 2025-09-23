
import cv2 as cv
import numpy as np
import time





def trackbar_callback(val):
    #print(val)
    pass




cap = cv.VideoCapture(1, cv.CAP_DSHOW)

#cap.set(cv.CAP_PROP_SETTINGS, 1)


finishline_camera = "Finishline Camera"
finishline_trackbar = "Finishline Trackbar"
finishline_canvas = "Finishline Canvas"
finishline_perspective = "Finishline Perspective"

cv.namedWindow(finishline_canvas, cv.WINDOW_NORMAL)
cv.namedWindow(finishline_perspective, cv.WINDOW_NORMAL)
cv.namedWindow(finishline_trackbar, cv.WINDOW_NORMAL)
cv.namedWindow(finishline_camera, cv.WINDOW_KEEPRATIO)

FRAME_HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
FRAME_WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))



cv.createTrackbar("TL_x", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("TL_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)

cv.createTrackbar("TR_x", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("TR_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)


cv.createTrackbar("BL_x", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("BL_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)

cv.createTrackbar("BR_x", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)
cv.createTrackbar("BR_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)


cv.createTrackbar("start_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)
cv.createTrackbar("start_w", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)


cv.createTrackbar("stop_y", finishline_trackbar, 0, FRAME_HEIGHT, trackbar_callback)
cv.createTrackbar("stop_w", finishline_trackbar, 0, FRAME_WIDTH, trackbar_callback)


while cap.isOpened():
    

    ret, frame = cap.read()


    if not ret:
        break


 
    tl_x = int(cv.getTrackbarPos("TL_x", finishline_trackbar))
    tl_y = int(cv.getTrackbarPos("TL_y", finishline_trackbar))

    tr_x = int(cv.getTrackbarPos("TR_x", finishline_trackbar))
    tr_y = int(cv.getTrackbarPos("TR_y", finishline_trackbar))

    bl_x = int(cv.getTrackbarPos("BL_x", finishline_trackbar))
    bl_y = int(cv.getTrackbarPos("BL_y", finishline_trackbar))

    br_x = int(cv.getTrackbarPos("BR_x", finishline_trackbar))
    br_y = int(cv.getTrackbarPos("BR_y", finishline_trackbar))


    start_line_y = int(cv.getTrackbarPos("start_y", finishline_trackbar))
    start_line_w = int(cv.getTrackbarPos("start_w", finishline_trackbar))

    stop_line_y = int(cv.getTrackbarPos("stop_y", finishline_trackbar))
    stop_line_w = int(cv.getTrackbarPos("stop_w", finishline_trackbar))



    #TODO: Delete after test

    #tl_x, tl_y = (200, 0)
    #tr_x, tr_y = (490, 0)
    #bl_x, bl_y = (200, 480)
    #br_x, br_y = (490, 480)
    #start_line_w, start_line_y = (131, 240)
    #stop_line_w, stop_line_y = (160, 240)



    
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
    

    if any([tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y, start_line_y, stop_line_y]):

        x_max = max(abs(tl_x - tr_x), abs(bl_x - br_x)) | 0
        y_max = max(abs(tl_y - bl_y), abs(tr_y - br_y)) | 0

        width = int(x_max)
        height = int(y_max)



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
        cv.line(perspective, (0, start_line_y), (start_line_w, start_line_y), (0, 0, 255), 2, cv.LINE_AA)
        cv.line(perspective, (stop_line_w, stop_line_y), (width, stop_line_y), (0, 0, 255), 2, cv.LINE_AA)

        cv.putText(perspective, f"Start:({start_line_y})", (0, start_line_y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        cv.putText(perspective, f"Stop:({stop_line_y})", (stop_line_w, stop_line_y-15), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow(finishline_perspective, perspective)

    

    cv.imshow(finishline_camera, frame)
    cv.imshow(finishline_canvas, canvas)
    


    key = cv.waitKey(1)


    if key & 0xFF == 27:
        break
    elif key == ord("c"):

        try:
            print("*"*20)
            #print(f"Top-Left: {tl_x, tl_y}")
            #print(f"Top_Right: {tr_x, tr_y}") 
            #print(f"Bottom-Left: {bl_x, bl_y}") 
            #print(f"Bottom-Right: {br_x, br_y}")
            print(f"Perspective Points: ({tl_x, tl_y}, {tr_x, tr_y}, {bl_x, bl_y}, {br_x, br_y})")
            print(f"Start_Line: ({0, start_line_y}, {start_line_w, start_line_y})")
            print(f"Stop_Line: ({stop_line_w, stop_line_y}, {width, stop_line_y})")
            print(f"Area: [{min(tl_y, tr_y)}:{max(bl_y, br_y)}, {min(tl_x, bl_x)}:{max(tr_x, br_x)}]")
            #print(f"x_max: {x_max}, y_max: {y_max}")
            #print(f"width: {width} height: {height}")
            print("*"*20)
        except Exception as e:
            print(e)




    
cap.release()
cv.destroyAllWindows()