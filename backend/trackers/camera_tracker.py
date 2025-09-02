import cv2 as cv
import numpy as np
#import sqlite3
import time
from backend.utils import set_camera_perspective
from backend.constants import COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR
from backend.models.sphero_bolt import SpheroBold
from backend.detectors.detector import Detector


#db = sqlite3.connect("opencv.sqlite")

#cursor = db.cursor()


#cursor.execute("""CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY AUTOINCREMENT,frame BLOB)""")


#cursor.connection.commit()



#cap = cv.VideoCapture("./tests/sphero1.mp4")



path_cap = cv.VideoCapture(2, cv.CAP_DSHOW)
path_cap.set(cv.CAP_PROP_SETTINGS, 1)

finishline_cap = cv.VideoCapture(1, cv.CAP_DSHOW)
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


    processed_path_frame = set_camera_perspective(frame=path_frame, top_left=(201, 20), top_right=(351, 24), bottom_left=(183, 413), bottom_right=(347, 414))
    
    processed_finishline_frame = set_camera_perspective(frame=finishline_frame, top_left=(201, 0), top_right=(436, 0), bottom_left= (178, 408), bottom_right= (458, 404))


    #frame = frame[51:434, 281:444]
    #frame = frame[:, 170:500]

    #cv.imshow("Processed Path Frame", processed_path_frame)
    #cv.imshow("Processed Finishline Frame", processed_finishline_frame)
    

    background_mask = background_subtractor_KNN.apply(processed_finishline_frame)

    background = background_subtractor_KNN.getBackgroundImage()




    if green_sphero is None:
        green_sphero = SpheroBold(color="Green", username="User", path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, background=background)
    else:
        green_sphero.path_frame = processed_path_frame
        green_sphero.finishline_frame = processed_finishline_frame
        green_sphero.background = background


    if green_detector is None:
        green_detector = Detector(path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, finishline_y=340, sphero_bolt=green_sphero, path_min_radius=9, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55, debug=True, is_strict=False)
    else:
        green_detector.path_frame = processed_path_frame
        green_detector.finishline_frame = processed_finishline_frame



    if yellow_sphero is None:
        yellow_sphero = SpheroBold(color="Yellow", username="User", path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, background=background)
    else:
        yellow_sphero.path_frame = processed_path_frame
        yellow_sphero.finishline_frame = processed_finishline_frame
        yellow_sphero.background = background



    if yellow_detector is None:
        yellow_detector = Detector(path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, finishline_y=340, sphero_bolt=yellow_sphero,  path_min_radius=5, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        yellow_detector.path_frame = processed_path_frame
        yellow_detector.finishline_frame = processed_finishline_frame


    

    if red_sphero is None:
        red_sphero = SpheroBold(color="Red", username="User", path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, background=background)
    else:
        red_sphero.path_frame = processed_path_frame
        red_sphero.finishline_frame = processed_finishline_frame
        red_sphero.background = background


    if red_detector is None:
        red_detector = Detector(path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, finishline_y=340, sphero_bolt=red_sphero, path_min_radius=5, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        red_detector.path_frame = processed_path_frame
        red_detector.finishline_frame = processed_finishline_frame



    if blue_sphero is None:
        blue_sphero = SpheroBold(color="Blue", username="User", path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, background=background)
    else:
        blue_sphero.path_frame = processed_path_frame
        blue_sphero.finishline_frame = processed_finishline_frame
        blue_sphero.background = background



    if blue_detector is None:
        blue_detector = Detector(path_frame=processed_path_frame, finishline_frame=processed_finishline_frame, finishline_y=340, sphero_bolt=blue_sphero,  path_min_radius=5, path_max_radius=55, finishline_min_radius=9, finishline_max_radius=55)
    else:
        blue_detector.path_frame = processed_path_frame
        blue_detector.finishline_frame = processed_finishline_frame






    

    finishline_frame_yellow = yellow_detector.get_processed_finishline_frame()
    finishline_frame_green = green_detector.get_processed_finishline_frame()
    finishline_frame_red = red_detector.get_processed_finishline_frame()
    finishline_frame_blue = blue_detector.get_processed_finishline_frame()

    path_frame_yellow = yellow_detector.get_processed_path_frame()
    path_frame_green = green_detector.get_processed_path_frame()
    path_frame_red = red_detector.get_processed_path_frame()
    path_frame_blue = blue_detector.get_processed_path_frame()



    #cv.imshow("yellow_finishline_frame", finishline_frame_yellow)
    #cv.imshow("green_finishline_frame", finishline_frame_green)
    #cv.imshow("red_finishline_frame", finishline_frame_red)
    #cv.imshow("blue_finishline_frame", finishline_frame_blue)

    #cv.imshow("yellow_path_frame", path_frame_yellow)
    #cv.imshow("green_path_frame", path_frame_green)
    #cv.imshow("red_path_frame", path_frame_red)
    #cv.imshow("blue_path_frame", path_frame_blue)

    cv.imshow("yellow_sphero.canvas", yellow_sphero.canvas)
    cv.imshow("green_sphero.canvas", green_sphero.canvas)
    cv.imshow("red_sphero.canvas", red_sphero.canvas)
    cv.imshow("blue_sphero.canvas", blue_sphero.canvas)
    
    finishline_temp1 = cv.bitwise_or(finishline_frame_yellow, finishline_frame_green)
    finishline_temp2 = cv.bitwise_or(finishline_frame_red, finishline_frame_blue)
    finishline_total = cv.bitwise_or(finishline_temp1, finishline_temp2)
    cv.imshow("Finishline Total", finishline_total)

    path_temp1 = cv.bitwise_or(path_frame_yellow, path_frame_green)
    path_temp2 = cv.bitwise_or(path_frame_red, path_frame_blue)
    path_total = cv.bitwise_or(path_temp1, path_temp2)
    cv.imshow("Path Total", path_total)



    """
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


