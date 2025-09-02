import cv2 as cv
import numpy as np
#import sqlite3
import time
from backend.constants import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR
from backend.models.sphero_bolt import SpheroBold

from backend.detectors.detector import Detector

#db = sqlite3.connect("opencv.sqlite")

#cursor = db.cursor()


#cursor.execute("""CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY AUTOINCREMENT,frame BLOB)""")


#cursor.connection.commit()



#cap = cv.VideoCapture("./tests/sphero1.mp4")

path_cap = cv.VideoCapture(1, cv.CAP_DSHOW)
path_cap.set(cv.CAP_PROP_SETTINGS, 1)

finishline_cap = cv.VideoCapture(2, cv.CAP_DSHOW)
finishline_cap.set(cv.CAP_PROP_SETTINGS, 1)







#cv.namedWindow("camera", cv.WINDOW_KEEPRATIO)

#FRAME_WIDTH = cap.get(cv.CAP_PROP_FRAME_WIDTH)
#FRAME_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

#fps = cap.get(cv.CAP_PROP_FPS)
#print("fps", fps)

#nframes = cap.get(cv.CAP_PROP_FRAME_COUNT)
#print("nframes", nframes)

#start = time.time()

#idx = 1

#start_sec = 5_000
#stop_sec = 10_000


#cap.set(cv.CAP_PROP_POS_FRAMES, 250)
#cap.set(cv.CAP_PROP_POS_MSEC, start_sec)



background_subtractor_KNN = cv.createBackgroundSubtractorKNN(history=500, detectShadows=True)


yellow_sphero = None
green_sphero = None
red_sphero = None
blue_sphero = None


yellow_detector = None
green_detector = None
red_detector = None
blue_detector = None


#t0 = time.time()
while path_cap.isOpened() and finishline_cap.isOpened():

    path_ret, path_frame = path_cap.read()
    finishline_ret, finishline_frame = finishline_cap.read()
    

    if not path_ret:
        print("Path camera is not working")
        break

    if not finishline_ret:
        print("Finishline camera is not working")
        break


    cv.imshow("Orginal Path Frame", path_frame)
    cv.imshow("Orginal Finishline Frame", finishline_frame)
    

    #frame_id = int(cap.get(cv.CAP_PROP_POS_FRAMES))
    #pos_sec = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0
    #print(f"frame id: {frame_id}, pos sec: {pos_sec}, real: {(time.time()-start)/1000.0}")

    #tl_x, tl_y = (213, 28)
    #tr_x, tr_y = (456, 38)
    #bl_x, bl_y = (190, 415)
    #br_x, br_y = (496, 405)

    tl_x, tl_y = (222, 26)
    tr_x, tr_y = (448, 27)
    bl_x, bl_y = (181, 473)
    br_x, br_y = (468, 474)

    x_max = max(abs(tl_x - tr_x), abs(bl_x - br_x))
    y_max = max(abs(tl_y - bl_y), abs(tr_y - br_y))

    width = 76*4
    height = 113*4

    pts_src = np.array([[tl_x, tl_y], 
                        [tr_x, tr_y], 
                        [br_x, br_y],
                        [bl_x, bl_y]], dtype=np.float32)
        
    pts_dst = np.array([[0, 0], 
                        [width, 0], 
                        [width, height], 
                        [0, height]], dtype=np.float32)


    matrix = cv.getPerspectiveTransform(pts_src, pts_dst)
    path_frame = cv.warpPerspective(path_frame, matrix, (width, height))
    
    #frame = frame[51:434, 281:444]
    #frame = frame[:, 170:500]
    cv.imshow("Prosessed Frame", path_frame)
    

    background_mask = background_subtractor_KNN.apply(path_frame)

    background = background_subtractor_KNN.getBackgroundImage()




    if green_sphero is None:
        green_sphero = SpheroBold(color="Green", username="User", path_frame=path_frame, finishline_frame=path_frame, background=background)
    else:
        green_sphero.path_frame = path_frame
        green_sphero.finishline_frame = path_frame
        green_sphero.background = background


    if green_detector is None:
        green_detector = Detector(path_frame=path_frame, finishline_frame=path_frame, finishline_y=340, sphero_bolt=green_sphero, path_min_radius=10, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        green_detector.path_frame = path_frame
        green_detector.finishline_frame = path_frame



    if yellow_sphero is None:
        yellow_sphero = SpheroBold(color="Yellow", username="User", path_frame=path_frame, finishline_frame=path_frame, background=background)
    else:
        yellow_sphero.path_frame = path_frame
        yellow_sphero.finishline_frame = path_frame
        yellow_sphero.background = background



    if yellow_detector is None:
        yellow_detector = Detector(path_frame=path_frame, finishline_frame=path_frame, finishline_y=340, sphero_bolt=yellow_sphero,  path_min_radius=10, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        yellow_detector.path_frame = path_frame
        yellow_detector.finishline_frame = path_frame


    

    if red_sphero is None:
        red_sphero = SpheroBold(color="Red", username="User", path_frame=path_frame, finishline_frame=path_frame, background=background)
    else:
        red_sphero.path_frame = path_frame
        red_sphero.finishline_frame = path_frame
        red_sphero.background = background


    if red_detector is None:
        red_detector = Detector(path_frame=path_frame, finishline_frame=path_frame, finishline_y=340, sphero_bolt=red_sphero, path_min_radius=10, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        red_detector.path_frame = path_frame
        red_detector.finishline_frame = path_frame



    if blue_sphero is None:
        blue_sphero = SpheroBold(color="Blue", username="User", path_frame=path_frame, finishline_frame=path_frame, background=background)
    else:
        blue_sphero.path_frame = path_frame
        blue_sphero.finishline_frame = path_frame
        blue_sphero.background = background



    if blue_detector is None:
        blue_detector = Detector(path_frame=path_frame, finishline_frame=path_frame, finishline_y=340, sphero_bolt=blue_sphero,  path_min_radius=10, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        blue_detector.path_frame = path_frame
        blue_detector.finishline_frame = path_frame








    detect_frame_yellow = yellow_detector.get_processed_finishline_frame()

    detect_frame_green = green_detector.get_processed_finishline_frame()

    detect_frame_red = red_detector.get_processed_finishline_frame()

    detect_frame_blue = blue_detector.get_processed_finishline_frame()

    cv.imshow("yellow_detect_frame", detect_frame_yellow)
    cv.imshow("green_detect_frame", detect_frame_green)
    cv.imshow("red_detect_frame", detect_frame_red)
    cv.imshow("blue_detect_frame", detect_frame_blue)

    cv.imshow("yellow_sphero.canvas", yellow_sphero.canvas)
    cv.imshow("green_sphero.canvas", green_sphero.canvas)
    cv.imshow("red_sphero.canvas", red_sphero.canvas)
    cv.imshow("blue_sphero.canvas", blue_sphero.canvas)
    
    temp1 = cv.bitwise_or(detect_frame_yellow, detect_frame_green)
    temp2 = cv.bitwise_or(detect_frame_red, detect_frame_blue)
    total = cv.bitwise_or(temp1, temp2)
    cv.imshow("Total", total)



   

    """
    hsv = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)
    hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])
    yellow_mask = cv.inRange(hsv, COLOR_RANGES["Yellow"]["Lower"], COLOR_RANGES["Yellow"]["Upper"])
    yellow_frame = cv.bitwise_and(frame.copy(), frame.copy(), mask=yellow_mask)
    cv.imshow("yellow_frame", yellow_frame)



    yellow_median = cv.medianBlur(frame.copy(), 9)
    cv.imshow("yellow_median", yellow_median)

    hsv_median = cv.cvtColor(frame.copy(), cv.COLOR_BGR2HSV)
    hsv_median[:, :, 2] = cv.equalizeHist(hsv_median[:, :, 2])
    yellow_median_mask = cv.inRange(hsv_median, COLOR_RANGES["Yellow"]["Lower"], COLOR_RANGES["Yellow"]["Upper"])


    yellow_median_frame = cv.bitwise_and(frame.copy(), frame.copy(), mask=yellow_median_mask)
    cv.imshow("yellow_median_frame", yellow_median_frame)
   


    yellow_gray = cv.cvtColor(yellow_frame.copy(), cv.COLOR_BGR2GRAY)
    #cv.imshow("yellow_mask_gray", yellow_mask_gray)
    _, yellow_mask_thr = cv.threshold(yellow_gray, 127, 255, cv.THRESH_BINARY)
    cv.imshow("yellow_mask_thr", yellow_mask_thr)

    


    yellow_median_gray = cv.cvtColor(yellow_median_frame.copy(), cv.COLOR_BGR2GRAY)
    #cv.imshow("yellow_mask_gray", yellow_mask_gray)
    _, yellow_median_thr = cv.threshold(yellow_median_gray, 127, 255, cv.THRESH_BINARY)
    cv.imshow("yellow_median_thr", yellow_median_thr)


    #####



    gray = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
    #cv.imshow("gray", gray)
    canny_frame = cv.Canny(gray, 100, 150) 
    #cv.imshow("canny_frame", canny_frame)
    fgmask_KNN = fgbg_KNN.apply(frame.copy())
    _, thresh = cv.threshold(fgmask_KNN.copy(), 170, 255, cv.THRESH_BINARY)
    #cv.imshow("bachground mask thresh", thresh)

    


    frame_median = cv.medianBlur(frame.copy(), 15)
    #cv.imshow("frame_median", frame_median)
    gray_median = cv.cvtColor(frame_median.copy(), cv.COLOR_BGR2GRAY)
    #cv.imshow("gray_median", gray_median)
    canny_median = cv.Canny(gray_median, 100, 150)
    #cv.imshow("canny_median", canny_median)
    fgmask_KNN_median = fgbg_KNN.apply(frame_median.copy())
    #cv.imshow("fgmask_KNN_median", fgmask_KNN_median)
    #_, thresh_median = cv.threshold(fgmask_kNN_stack.copy(), 170, 255, cv.THRESH_BINARY)
    _, fgmask_KNN_thresh_median = cv.threshold(cv.medianBlur(fgmask_KNN_median.copy(), 15), 170, 255, cv.THRESH_BINARY)
    #cv.imshow("fgmask_KNN_thresh_median", fgmask_KNN_thresh_median)



    background = fgbg_KNN.getBackgroundImage()

    #cv.imshow("backgroung", background)


    #cv.imwrite(f"frames/frame{idx}.jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    #cv.imwrite(f"frames/frame{idx}.png", frame, [cv.IMWRITE_PNG_COMPRESSION, 0])
    

    #elapsed = time.time() - t0
    #target = cap.get(cv.CAP_PROP_POS_MSEC)/1000.0

    #delay = target - elapsed

    
    #time.sleep(delay*2)
    #idx += 1
    #time.sleep(1.0/fps)
    """


    if cv.waitKey(1) & 0xFF == 27:
        break




#cursor.close()
#db.close()

path_cap.release()
cv.destroyAllWindows()

#stop = time.time()
#print(stop-start)


