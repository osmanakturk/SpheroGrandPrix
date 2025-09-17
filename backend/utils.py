import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from enum import Enum
from dataclasses import dataclass
from backend.constants import HSV_RANGES_MANUAL, HSV_RANGES_NORMAL, HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_BGR, COLORS_HSV






class CaptureApi(Enum):
    Windows = cv.CAP_DSHOW
    Linux = cv.CAP_V4L2
    Mac = cv.CAP_AVFOUNDATION



class HsvColorsRange(Enum):
    NORMAL = HSV_RANGES_NORMAL
    WIDE = HSV_RANGES_WIDE
    STRICT = HSV_RANGES_STRICT
    MANUAL = HSV_RANGES_MANUAL




class SpheroColor(Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    YELLOW = "Yellow"




@dataclass
class CameraConfig:
    cap_api: CaptureApi 
    cap_index: Optional[int] = None 
    cap_source: Optional[str] = None 
    cap_width: Optional[int] = 640 
    cap_height: Optional[int] = 480 
    cap_fps: Optional[int] = 30  
    perspective_top_left: Optional[Tuple[int, int]] = None 
    perspective_top_right: Optional[Tuple[int, int]] = None 
    perspective_bottom_left: Optional[Tuple[int, int]] = None 
    perspective_bottom_right: Optional[Tuple[int, int]] = None 
    perspective_width: Optional[int] = None 
    perspective_height: Optional[int] = None 
    start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None 
    finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None




def set_camera_perspective(frame: cv.typing.MatLike, 
                           top_left: Tuple[int, int], 
                           top_right: Tuple[int, int], 
                           bottom_left: Tuple[int, int], 
                           bottom_right: Tuple[int, int], 
                           width: Optional[int] = None, 
                           height: Optional[int] = None, 
                           start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
                           finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
                           ) -> cv.typing.MatLike:
    
    if frame is None:
        print("Frame is None")
        return
    
    if not (top_left and top_right and bottom_left and bottom_right):
        print("Perspective corners must all be provided")
        return
    
    tl_x, tl_y = top_left
    tr_x, tr_y = top_right
    bl_x, bl_y = bottom_left
    br_x, br_y = bottom_right

    if width is None or height is None:
        x_max = max(abs(tl_x - tr_x), abs(bl_x - br_x))
        y_max = max(abs(tl_y - bl_y), abs(tr_y - br_y))

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

    processed_frame = cv.warpPerspective(frame, matrix, (width, height))

    if start_line is not None and finish_line is not None:
        cv.line(processed_frame, start_line[0], start_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)
        cv.line(processed_frame, finish_line[0], finish_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)

    return processed_frame
