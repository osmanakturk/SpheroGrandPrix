import cv2 as cv
import numpy as np
#import sqlite3
import time
from backend.utils import set_camera_perspective, HsvColorsRange
from backend.models.sphero_bolt import SpheroBolt
from backend.detectors.detector import Detector
from backend.models.lap import Lap





def main():
        
    #db = sqlite3.connect("opencv.sqlite")

    #cursor = db.cursor()


    #cursor.execute("""CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY AUTOINCREMENT,frame BLOB)""")


    #cursor.connection.commit()



    #cap = cv.VideoCapture("./tests/sphero1.mp4")

    background = cv.imread("paths/background.png", cv.IMREAD_COLOR)

    path_cap = cv.VideoCapture(2, cv.CAP_DSHOW)
    path_cap.set(cv.CAP_PROP_SETTINGS, 1)

    finishline_cap = cv.VideoCapture(1, cv.CAP_DSHOW)
    finishline_cap.set(cv.CAP_PROP_SETTINGS, 1)

    lap = None

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



    #background_subtractor_KNN = cv.createBackgroundSubtractorKNN(history=500,dist2Threshold=400.0, detectShadows=True)




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
        


        processed_path_frame = set_camera_perspective(
            frame=path_frame, 
            top_left=(200, 0), 
            top_right=(370, 0), 
            bottom_left=(200, 480), 
            bottom_right=(370, 480)
            )
        

        processed_finishline_frame = set_camera_perspective(
            frame=finishline_frame, 
            top_left=(200, 0), 
            top_right=(490, 0), 
            bottom_left=(200, 480), 
            bottom_right=(490, 480)
            )


        if lap is None:
            lap = Lap(path_frame=processed_path_frame, 
                      finishline_frame=processed_finishline_frame, 
                      background_img=background, 
                      is_started=True
                      )
        else:
            lap.path_frame = processed_path_frame
            lap.finishline_frame = processed_finishline_frame
            returned_path_frame = lap.get_processed_path_frame(
                hsv_ranges=HsvColorsRange.NORMAL, 
                min_radius=15, 
                max_radius=35, 
                bilateral_diameter=9, 
                bilateral_sigma_color=75, 
                bilateral_sigma_space=75, 
                median_kernel_size=9, 
                clahe_clip_limit=4, 
                clahe_tile_grid_size=9, 
                morph_kernel_size=5, 
                morph_iterator=1, 
                contours_chain_approx_simple=True
                )
            
            returner_finishline_frame = lap.get_processed_finishline_frame(
                hsv_ranges=HsvColorsRange.NORMAL, 
                min_radius=15, 
                max_radius=35, 
                start_line=((0, 240), (131, 240)), 
                finish_line=((160, 240), (290, 240)), 
                bilateral_diameter=9,
                bilateral_sigma_color=75,
                bilateral_sigma_space=75,
                median_kernel_size=9,
                clahe_clip_limit=4,
                clahe_tile_grid_size=9,
                morph_kernel_size=5,
                morph_iterator=1,
                contours_chain_approx_simple=True
                )
            
            cv.imshow("returned_path_frame", returned_path_frame)
            cv.imshow("returner_finishline_frame", returner_finishline_frame)


        cv.imshow("canvas", lap.sphero_bolt_yellow.canvas)
 
        cv.imshow("Processed Path Frame", processed_path_frame)
        cv.imshow("Processed Finishline Frame", processed_finishline_frame)
        cv.imshow("Background", background)
        



        if cv.waitKey(1) & 0xFF == 27:
            break




    #cursor.close()
    #db.close()

    path_cap.release()
    finishline_cap.release()
    cv.destroyAllWindows()

    #stop = time.time()
    #print(stop-start)



if __name__=="__main__":
    main()

