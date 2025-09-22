from typing import Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from backend.enums import CaptureApi, HsvColorsRange, SpheroColor
import cv2 as cv
from datetime import datetime

if TYPE_CHECKING:
      from backend.models.sphero_bolt import SpheroBolt




@dataclass
class CameraConfig:
    cap_api: Optional[CaptureApi] = None
    cap_index: Optional[int] = None 
    cap_source: Optional[str] = None 
    cap_width: Optional[int] = 640 
    cap_height: Optional[int] = 480 
    cap_fps: Optional[int] = 30  
    perspective_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None
    perspective_width: Optional[int] = None 
    perspective_height: Optional[int] = None 
    start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None 
    finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None




@dataclass
class DetectorConfig:
    sphero_bolt: Optional["SpheroBolt"] = None
    hsv_ranges: HsvColorsRange = HsvColorsRange.NORMAL 
    min_radius: Optional[int] = None 
    max_radius: Optional[int] = None 
    bilateral_diameter: int = 9 
    bilateral_sigma_color: int = 75 
    bilateral_sigma_space: int = 75 
    median_kernel_size: int = 9 
    clahe_clip_limit: float = 4.0 
    clahe_tile_grid_size: int = 9 
    morph_kernel_size: int = 5 
    morph_iterator: int = 1 
    contours_chain_approx_simple: bool = True 
    debug: bool = False


@dataclass
class SpheroConfig:
    lap_id: str 
    color:  SpheroColor 
    path_frame: Optional[cv.typing.MatLike] = None 
    finishline_frame: Optional[cv.typing.MatLike] = None 
    background: Optional[cv.typing.MatLike] = None 
    is_lap_started: bool = False 
    is_lap_stopped: bool = False 
    debug: bool = False