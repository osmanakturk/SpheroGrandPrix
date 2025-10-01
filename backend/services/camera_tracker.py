import cv2 as cv
import numpy as np
import time, os, sys
from backend.enums import CaptureApi, HsvColorsRange
from backend.configs import CameraConfig, DetectorConfig
from backend.models.lap import Lap
from backend.models.camera import Camera
from typing import Optional, Tuple
import threading


LOCK = threading.Lock()
GLOBAL_LAB = Lap()

GLOBAL_USERNAME_RED = "Red"
GLOBAL_USERNAME_YELLOW = "Yellow"
GLOBAL_USERNAME_BLUE = "Blue"
GLOBAL_USERNAME_GREEN = "Green"




STATUS_CAP: Optional[Camera] = None
FINISHLINE_CAP: Optional[Camera] = None




def start_tracker(status_cap_config:CameraConfig, 
                  finishline_cap_config:CameraConfig
                  ) -> bool:

    global STATUS_CAP, FINISHLINE_CAP


    
    with LOCK:
        try:
            STATUS_CAP = Camera(config=status_cap_config)
            status_ret =  STATUS_CAP.open()
            print(f"Status Cap Open: {status_ret}")
        except Exception as e:
            print(f"Status Cap Open: {e}")
        

        while not status_ret:
            try:
                print("Trying to Open Status Cap")
                STATUS_CAP.release()
                status_ret = STATUS_CAP.open()
                print(f"Status Cap Open: {status_ret}")
            except Exception as e:
                print(f"Status Cap Open: {e}")

        try:
            FINISHLINE_CAP = Camera(config=finishline_cap_config)
            finishline_ret = FINISHLINE_CAP.open()
            print(f"Finishline Cap Open: {finishline_ret}")
        except Exception as e:
            print(f"Finishline Cap Open: {e}")

        while not finishline_ret:
            try:
                print("Trying to Open Finishline Cap")
                FINISHLINE_CAP.release()
                finishline_ret = FINISHLINE_CAP.open()
                print(f"Finishline Cap Open: {finishline_ret}")
            except Exception as e:
                print(f"Finishline Cap Open: {e}")
    return True
    



def get_tracker(
        finishline_detector_config: DetectorConfig, 
        back_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None, 
        middle_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None, 
        front_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None, 
        debug: bool = False, 
        ) -> Optional[Tuple[bytes, bytes, Lap]]:
        

    global GLOBAL_LAB, STATUS_CAP, FINISHLINE_CAP
    lap = GLOBAL_LAB
    
    with LOCK:

        try:
            ok_status = STATUS_CAP.read()
            #print(f"Status Cap Read: {ok_status}")
        except Exception as e:
            print(f"Status Cap Read: {e}")


        while not ok_status:
            try:
                STATUS_CAP.release()
                time.sleep(0.2)
                STATUS_CAP.open()
                ok_status = STATUS_CAP.read()
                print(f"Status Cap Read: {ok_status}")
            except Exception as e:
                print(f"Status Cap Read: {e}")
        
        try:
            ok_fin = FINISHLINE_CAP.read()
            #print(f"Finishline Cap Read: {ok_fin}")
        except Exception as e:
            print(f"Finishline Cap Read: {e}")
    

        while not ok_fin:
            try:
                FINISHLINE_CAP.release()
                time.sleep(0.2)
                FINISHLINE_CAP.open()
                ok_fin = FINISHLINE_CAP.read()
                print(f"Finishline Cap Read: {ok_fin}")
            except Exception as e:
                print(f"Finishline Cap Read: {e}")

    if debug:
        cv.imshow("Original Status Frame", STATUS_CAP.frame)
        cv.imshow("Original Finishline Frame", FINISHLINE_CAP.frame)


    
    FINISHLINE_CAP.set_perspective_frame()
    
    status_frame = STATUS_CAP.frame.copy()
    perspectived_finishline_frame = FINISHLINE_CAP.perspective_frame.copy()


    if debug:
        cv.imshow("Perspectived Finishline Frame", perspectived_finishline_frame)
    
    #_, processed_path_frame_jpg_encode = cv.imencode(".jpg", processed_path_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    #_, processed_finishline_frame_jpg_encode = cv.imencode(".jpg", processed_finishline_frame, [cv.IMWRITE_JPEG_QUALITY, 100])
    #processed_path_frame_decode = cv.imdecode(processed_path_frame_jpg_encode, cv.IMREAD_COLOR)
    #processed_finishline_frame_decode = cv.imdecode(processed_finishline_frame_jpg_encode, cv.IMREAD_COLOR)
    
    
    returner_finishline_frame = None
    
    if lap.is_started:
        
        lap.finishline_frame = FINISHLINE_CAP.perspective_frame
   
        
        returner_finishline_frame = lap.get_processed_finishline_frame(
            config=finishline_detector_config, 
            start_line=FINISHLINE_CAP.start_line, 
            finish_line=FINISHLINE_CAP.finish_line
        )

        if debug:
            
            cv.imshow("Returner Finishline Frame", returner_finishline_frame)
  
    

   
    back_points = np.array(back_points)
    middle_points = np.array(middle_points)
    front_points = np.array(front_points)
    overlay = status_frame.copy()
    

    if back_points is not None:
        cv.fillPoly(overlay, [back_points], (255, 0, 0))


    if middle_points is not None:
        if lap.is_started: 
            cv.fillPoly(overlay, [middle_points], (0, 255, 0))
        else:
            cv.fillPoly(overlay, [middle_points], (0, 0, 255))
        

    if front_points is not None:
        cv.fillPoly(overlay, [front_points], (255, 0, 0))
        

    status_result = cv.addWeighted(status_frame, 0.5, overlay, 0.5, 0)
    cv.polylines(status_result, [back_points, middle_points, front_points], True, (0,0,0), 1, cv.LINE_AA)


    
    if returner_finishline_frame is not None:
        finishline_result = cv.bitwise_or(perspectived_finishline_frame, returner_finishline_frame) 
    else: 
        cv.putText(perspectived_finishline_frame, "START", (perspectived_finishline_frame.shape[1]//4, perspectived_finishline_frame.shape[0]//2), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(perspectived_finishline_frame, "GAME", (perspectived_finishline_frame.shape[1]//4, perspectived_finishline_frame.shape[0]//2+30), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        finishline_result = perspectived_finishline_frame
    

   

    if debug:
        cv.imshow("Status Result", status_result)
        cv.imshow("Finishline Result", finishline_result)

    status_ok, status_jpg = cv.imencode(".jpg", status_result, [cv.IMWRITE_JPEG_QUALITY, 100])
    finishline_ok, finishline_jpg = cv.imencode(".jpg", finishline_result, [cv.IMWRITE_JPEG_QUALITY, 100])

    result1 = None
    result2 = None

    if status_ok:
        result1 = status_jpg.tobytes()

    if finishline_ok:
        result2 = finishline_jpg.tobytes()

    return(result1, result2, GLOBAL_LAB)




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




def change_username_red(username: str) -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_RED

    GLOBAL_USERNAME_RED = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_red.username = GLOBAL_USERNAME_RED

    return True


def change_username_yellow(username: str) -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_YELLOW

    GLOBAL_USERNAME_YELLOW = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_yellow.username = GLOBAL_USERNAME_YELLOW
    
    return True

def change_username_blue(username: str) -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_BLUE

    GLOBAL_USERNAME_BLUE = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_blue.username = GLOBAL_USERNAME_BLUE

    return True



def change_username_green(username: str) -> bool:
    global GLOBAL_LAB, GLOBAL_USERNAME_GREEN

    GLOBAL_USERNAME_GREEN = username

    if GLOBAL_LAB.is_started:
        GLOBAL_LAB.sphero_bolt_green.username = GLOBAL_USERNAME_GREEN

    return True




def release_all():
    global STATUS_CAP, FINISHLINE_CAP

    try:
        STATUS_CAP.release()
        FINISHLINE_CAP.release()
        cv.destroyAllWindows()
        print("All Cameras Released")
    except Exception as e:
        print(f"Release All Failed: {e}")


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

    def trackbar_reset_red(val):
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



    def trackbar_reset_yellow(val):
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
    
    
    def trackbar_reset_blue(val):
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
    
    
    def trackbar_reset_green(val):
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
    cv.createTrackbar("reset_red", trackbar, 0, 1, trackbar_reset_red)
    cv.createTrackbar("reset_yellow", trackbar, 0, 1, trackbar_reset_yellow)
    cv.createTrackbar("reset_blue", trackbar, 0, 1, trackbar_reset_blue)
    cv.createTrackbar("reset_green", trackbar, 0, 1, trackbar_reset_green)



    start_tracker(
        finishline_cap_config=CameraConfig(
            cap_api=CaptureApi.Windows, 
            cap_index=2, 
            perspective_top_left=(200, 0), 
            perspective_top_right=(490, 0), 
            perspective_bottom_left=(200, 480), 
            perspective_bottom_right=(490, 480), 
            start_line=((0, 240), (131, 240)), 
            finish_line=((160, 240), (290, 240))
            ), 

        status_cap_config=CameraConfig(
            cap_api=CaptureApi.Windows, 
            cap_index=1
            )

    )

    while True:
        status_buf, finishline_buf, lap = get_tracker(
            back_points=((296, 65), (353, 381), (395, 90)), 
            middle_points=((296, 65), (226, 120), (246, 366), (460, 396), (440, 140)), 
            front_points= ((296, 65), (310, 480), (353, 381)), 
            finishline_detector_config= DetectorConfig(
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
                ),
            debug=False)

        if status_buf is not None:
            status_arr = np.frombuffer(status_buf, np.uint8)
            status_img = cv.imdecode(status_arr, cv.IMREAD_COLOR)

            if status_img is not None:
                cv.imshow("Status", status_img)

        if finishline_buf is not None:
            finishline_arr = np.frombuffer(finishline_buf, np.uint8)
            finishline_img = cv.imdecode(finishline_arr, cv.IMREAD_COLOR)

            if finishline_img is not None:
                cv.imshow("Finishline", finishline_img)
        



        if cv.waitKey(1) & 0xFF == 27:
            break
    
    release_all()

