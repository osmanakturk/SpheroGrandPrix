import cv2 as cv
import numpy as np
import time, os, sys
from backend.utils import HsvColorsRange
from backend.models.lap import Lap
from backend.models.camera import CaptureApi, Camera
from typing import Optional, Tuple

GLOBAL_LAB = Lap()

GLOBAL_USERNAME_RED = "Red"
GLOBAL_USERNAME_YELLOW = "Yellow"
GLOBAL_USERNAME_BLUE = "Blue"
GLOBAL_USERNAME_GREEN = "Green"








BACKGROUND = None
PATH_CAP: Optional[Camera] = None
FINISHLINE_CAP: Optional[Camera] = None




def start_tracker() -> bool:

    global BACKGROUND, PATH_CAP, FINISHLINE_CAP

    BACKGROUND = cv.imread("paths/background.png", cv.IMREAD_COLOR)
    
    
    PATH_CAP = Camera(cap_api=CaptureApi.Windows, 
                      cap_index=2, 
                      perspective_top_left=(200, 0), 
                      perspective_top_right=(370, 0), 
                      perspective_bottom_left=(200, 480), 
                      perspective_bottom_right=(370, 480)
                      )
    
    path_ret =  PATH_CAP.open()
    
    while not path_ret:
        PATH_CAP.release()
        
        path_ret = PATH_CAP.open()
        
    
    FINISHLINE_CAP = Camera(cap_api=CaptureApi.Windows, 
                            cap_index=1, 
                            perspective_top_left=(200, 0), 
                            perspective_top_right=(490, 0), 
                            perspective_bottom_left=(200, 480), 
                            perspective_bottom_right=(490, 480), 
                            start_line=((0, 240), (131, 240)), 
                            finish_line=((160, 240), (290, 240))
                            )
    
    
    
    finishline_ret = FINISHLINE_CAP.open()
    
    while not finishline_ret:
        FINISHLINE_CAP.release()
        
        finishline_ret = FINISHLINE_CAP.open()

    return True
    





def lap_start() -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_RED, GLOBAL_USERNAME_YELLOW, GLOBAL_USERNAME_BLUE, GLOBAL_USERNAME_GREEN

    if not GLOBAL_LAB.is_started:

        GLOBAL_LAB = Lap.start()

        GLOBAL_LAB.sphero_bolt_red.username = GLOBAL_USERNAME_RED
        GLOBAL_LAB.sphero_bolt_yellow.username = GLOBAL_USERNAME_YELLOW
        GLOBAL_LAB.sphero_bolt_blue.username = GLOBAL_USERNAME_BLUE
        GLOBAL_LAB.sphero_bolt_green.username = GLOBAL_USERNAME_GREEN

        return True
    
    return False


def lap_stop() -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_RED, GLOBAL_USERNAME_YELLOW, GLOBAL_USERNAME_BLUE, GLOBAL_USERNAME_GREEN

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.stop()
        GLOBAL_USERNAME_RED = "Red"
        GLOBAL_USERNAME_YELLOW = "Yellow"
        GLOBAL_USERNAME_BLUE = "Blue"
        GLOBAL_USERNAME_GREEN = "Green"
        return True
    
    return False



def reset_red() -> bool:
    global GLOBAL_LAB

    if GLOBAL_LAB.is_started:
        result = GLOBAL_LAB.sphero_bolt_red.reset()
        return result
    
    return False



def reset_yellow() -> bool:
    global GLOBAL_LAB

    if GLOBAL_LAB.is_started:
        result = GLOBAL_LAB.sphero_bolt_yellow.reset()
        return result
    
    return False



def reset_blue() -> bool:
    global GLOBAL_LAB

    if GLOBAL_LAB.is_started:
        result = GLOBAL_LAB.sphero_bolt_blue.reset()
        return result
    
    return False




def reset_green() -> bool:
    global GLOBAL_LAB

    if GLOBAL_LAB.is_started:
        result = GLOBAL_LAB.sphero_bolt_green.reset()
        return result
    
    return False




def change_username_red(username: str):
    global GLOBAL_LAB, GLOBAL_USERNAME_RED

    GLOBAL_USERNAME_RED = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_red.username = GLOBAL_USERNAME_RED


def change_username_yellow(username: str):
    global GLOBAL_LAB, GLOBAL_USERNAME_YELLOW

    GLOBAL_USERNAME_YELLOW = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_yellow.username = GLOBAL_USERNAME_YELLOW


def change_username_blue(username: str):
    global GLOBAL_LAB, GLOBAL_USERNAME_BLUE

    GLOBAL_USERNAME_BLUE = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_blue.username = GLOBAL_USERNAME_BLUE


def change_username_green(username: str):
    global GLOBAL_LAB, GLOBAL_USERNAME_GREEN

    GLOBAL_USERNAME_GREEN = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_green.username = GLOBAL_USERNAME_GREEN




def get_tracker(debug: bool = False) -> Optional[Tuple[bytes, bytes, Lap]]:
        

    global GLOBAL_LAB, PATH_CAP, FINISHLINE_CAP
    lap = GLOBAL_LAB
    

    ok_path = PATH_CAP.read()
    
  
    while not ok_path:
        print(f"Path r: {ok_path}")
        PATH_CAP.release()
        time.sleep(0.2)
        PATH_CAP.open()
        ok_path = PATH_CAP.read()

    
    ok_fin = FINISHLINE_CAP.read()
    

    while not ok_fin:
        print(f"Finishline r: {ok_fin}")
        FINISHLINE_CAP.release()
        time.sleep(0.2)
        FINISHLINE_CAP.open()
        ok_fin = FINISHLINE_CAP.read()


    if debug:
        cv.imshow("Original Path Frame", PATH_CAP.frame)
        cv.imshow("Original Finishline Frame", FINISHLINE_CAP.frame)


    PATH_CAP.set_perspective_frame()
    FINISHLINE_CAP.set_perspective_frame()
    perspectived_path_frame = PATH_CAP.perspective_frame.copy()
    perspectived_finishline_frame = FINISHLINE_CAP.perspective_frame.copy()


    if debug:
        cv.imshow("Perspectived Path Frame", perspectived_path_frame)
        cv.imshow("Perspectived Finishline Frame", perspectived_finishline_frame)
    
    #_, processed_path_frame_jpg_encode = cv.imencode(".jpg", processed_path_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    #_, processed_finishline_frame_jpg_encode = cv.imencode(".jpg", processed_finishline_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    #processed_path_frame_decode = cv.imdecode(processed_path_frame_jpg_encode, cv.IMREAD_COLOR)
    #processed_finishline_frame_decode = cv.imdecode(processed_finishline_frame_jpg_encode, cv.IMREAD_COLOR)
    
    returned_path_frame = None
    returner_finishline_frame = None
    
    if lap.is_started:
        lap.path_frame = PATH_CAP.perspective_frame
        lap.finishline_frame = FINISHLINE_CAP.perspective_frame
        lap.background_img = BACKGROUND
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
        
        if debug:
            cv.imshow("Returned Path Frame", returned_path_frame)
            cv.imshow("Returner Finishline Frame", returner_finishline_frame)
            cv.imshow("Canvas Red", lap.sphero_bolt_red.canvas)
            cv.imshow("Canvas Yellow", lap.sphero_bolt_yellow.canvas)
            cv.imshow("Canvas Blue", lap.sphero_bolt_blue.canvas)
            cv.imshow("Canvas Green", lap.sphero_bolt_green.canvas)
            #cv.imshow("Background", background)
    
    
    path_result = cv.bitwise_or(perspectived_path_frame, returned_path_frame) if returned_path_frame is not None else perspectived_path_frame
    finishline_result = cv.bitwise_or(perspectived_finishline_frame, returner_finishline_frame) if returner_finishline_frame is not None else perspectived_finishline_frame
    

   

    if debug:
        cv.imshow("Path Result", path_result)
        cv.imshow("Finishline Result", finishline_result)

    path_ok, path_jpg = cv.imencode(".jpg", path_result, [cv.IMWRITE_JPEG_QUALITY, 100])
    finishline_ok, finishline_jpg = cv.imencode(".jpg", finishline_result, [cv.IMWRITE_JPEG_QUALITY, 100])

    result1 = None
    result2 = None

    if path_ok:
        result1 = path_jpg.tobytes()

    if finishline_ok:
        result2 = finishline_jpg.tobytes()

    return(result1, result2, GLOBAL_LAB)


def release_all():
    PATH_CAP.release()
    FINISHLINE_CAP.release()
    cv.destroyAllWindows()


if __name__=="__main__":

    trackbar = "Trackbar"
    
    def trackbar_lap_stop(val):
        if val == 1:
            lap_stop()

        cv.setTrackbarPos("stop_lap", trackbar, 0)


    def trackbar_lap_start(val):
        if val == 1:
            lap_start()

        cv.setTrackbarPos("start_lap", trackbar, 0)

    def reset_red(val):
        if val != 1:
            return
        
        if not GLOBAL_LAB.is_started:
            cv.setTrackbarPos("reset_red", trackbar, 0)
            print("Lap not started yet")
            return
        
        ok = GLOBAL_LAB.sphero_bolt_red.reset()

        if ok:
            print(f"Red resetted with id: {GLOBAL_LAB.sphero_bolt_red.id}")
        else:
            print(f"Red not resetted with id: {GLOBAL_LAB.sphero_bolt_red.id}")

        cv.setTrackbarPos("reset_red", trackbar, 0)



    def reset_yellow(val):
        if val != 1:
            return
        
        if not GLOBAL_LAB.is_started:
            cv.setTrackbarPos("reset_yellow", trackbar, 0)
            print("Lap not started yet")
            return
        
        ok = GLOBAL_LAB.sphero_bolt_yellow.reset()

        if ok:
            print(f"Yellow resetted with id: {GLOBAL_LAB.sphero_bolt_yellow.id}")
        else:
            print(f"Yellow not resetted with id: {GLOBAL_LAB.sphero_bolt_yellow.id}")

        cv.setTrackbarPos("reset_yellow", trackbar, 0)
    
    
    def reset_blue(val):
        if val != 1:
            return
        
        if not GLOBAL_LAB.is_started:
            cv.setTrackbarPos("reset_blue", trackbar, 0)
            print("Lap not started yet")
            return
        
        ok = GLOBAL_LAB.sphero_bolt_blue.reset()

        if ok:
            print(f"Blue resetted with id: {GLOBAL_LAB.sphero_bolt_blue.id}")
        else:
            print(f"Blue not resetted with id: {GLOBAL_LAB.sphero_bolt_blue.id}")

        cv.setTrackbarPos("reset_blue", trackbar, 0)
    
    
    def reset_green(val):
        if val != 1:
            return
        
        if not GLOBAL_LAB.is_started:
            cv.setTrackbarPos("reset_green", trackbar, 0)
            print("Lap not started yet")
            return
        
        ok = GLOBAL_LAB.sphero_bolt_green.reset()

        if ok:
            print(f"Green resetted with id: {GLOBAL_LAB.sphero_bolt_green.id}")
        else:
            print(f"Green not resetted with id: {GLOBAL_LAB.sphero_bolt_green.id}")

        cv.setTrackbarPos("reset_green", trackbar, 0)


    cv.namedWindow(trackbar, cv.WINDOW_NORMAL)
    cv.createTrackbar("start_lap", trackbar, 0, 1, trackbar_lap_start)
    cv.createTrackbar("stop_lap", trackbar, 0, 1, trackbar_lap_stop)
    cv.createTrackbar("reset_red", trackbar, 0, 1, reset_red)
    cv.createTrackbar("reset_yellow", trackbar, 0, 1, reset_yellow)
    cv.createTrackbar("reset_blue", trackbar, 0, 1, reset_blue)
    cv.createTrackbar("reset_green", trackbar, 0, 1, reset_green)


    start_tracker()

    while True:
        path_buf, finishline_buf, lap = get_tracker(debug=False)

        if path_buf is not None:
            path_arr = np.frombuffer(path_buf, np.uint8)
            path_img = cv.imdecode(path_arr, cv.IMREAD_COLOR)

            if path_img is not None:
                cv.imshow("Path", path_img)

        if finishline_buf is not None:
            finishline_arr = np.frombuffer(finishline_buf, np.uint8)
            finishline_img = cv.imdecode(finishline_arr, cv.IMREAD_COLOR)

            if finishline_img is not None:
                cv.imshow("Finishline", finishline_img)
        



        if cv.waitKey(1) & 0xFF == 27:
            break
    
    release_all()

