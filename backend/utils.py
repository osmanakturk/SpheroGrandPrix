import cv2 as cv
import numpy as np
from typing import Tuple, Optional
from enum import Enum
from .constants import HSV_RANGES_MANUAL, HSV_RANGES_NORMAL, HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_BGR, COLORS_HSV


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





def set_camera_perspective(frame: cv.typing.MatLike, 
                           top_left: Tuple[int, int], 
                           top_right: Tuple[int, int], 
                           bottom_left: Tuple[int, int], 
                           bottom_right: Tuple[int, int], 
                           width: Optional[int] = None, 
                           height: Optional[int] = None) -> cv.typing.MatLike:
    
    
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

    return cv.warpPerspective(frame, matrix, (width, height))
