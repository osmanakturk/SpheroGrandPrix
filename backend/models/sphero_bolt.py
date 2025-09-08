import uuid
import time
import numpy as np
import cv2 as cv
from datetime import datetime
from backend.constants import PATH, HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_HSV, COLORS_BGR
from typing import Optional, Tuple
from backend.utils import HsvColorsRange, SpheroColor




class SpheroBolt():

    def __init__(self,
                 lap_id: str,
                 color:  SpheroColor, 
                 username: Optional[str] = None,
                 path_frame: Optional[cv.typing.MatLike] = None, 
                 finishline_frame: Optional[cv.typing.MatLike] = None,
                 background: Optional[cv.typing.MatLike] = None, 
                 start_time: Optional[datetime] = None, 
                 finish_time: Optional[datetime] = None, 
                 total_lap_time: Optional[float] = None,
                 path_previous_center: Optional[Tuple[int, int]] = None, 
                 path_center: Optional[Tuple[int, int]] = None, 
                 path_radius: Optional[int] = None,
                 finishline_center: Optional[Tuple[int, int]] = None,
                 finishline_previous_center: Optional[Tuple[int, int]] = None,
                 finishline_radius: Optional[int] = None,
                 is_started: bool = False,
                 is_finished: bool = False, 
                 is_lap_started: bool = False,
                 is_lap_stopped: bool = False,
                 debug: bool = False
                 ):
        

        self._lap_id = lap_id
        self._color = color
        self._id = f"{lap_id}_{self._color.value}"
        self._username = username if username is not None else self._color.value
        self._background = background.copy() if background is not None else None
        self._is_started = is_started
        self._is_finished = is_finished
        self._is_lap_started = is_lap_started
        self._is_lap_stopped = is_lap_stopped
        self._start_time = start_time
        self._finish_time = finish_time
        self._total_lap_time = total_lap_time
        self._path_previous_center = path_previous_center
        self._path_center = path_center
        self._path_radius = path_radius
        self._finishline_center = finishline_center
        self._finishline_previous_center = finishline_previous_center
        self._finishline_radius = finishline_radius
        self._path_frame = path_frame.copy() if path_frame is not None else None
        self._finishline_frame = finishline_frame.copy() if finishline_frame is not None else None
        self._canvas = np.zeros_like(self._path_frame, np.uint8) if self._path_frame is not None else None
        self._debug = debug
  


    @property
    def lap_id(self) -> str:
        return self._lap_id

    @lap_id.setter
    def lap_id(self, lap_id: str) -> None:
        self._lap_id = lap_id


    @property
    def id(self) -> str:
        return f"{self._lap_id}_{self._color.value}"
    
    @id.setter
    def id(self, lap_id: str) -> None:
        self._id = f"{lap_id}_{self._color.value}"

    @property
    def path_frame(self) -> Optional[cv.typing.MatLike]:
        return self._path_frame


    @path_frame.setter
    def path_frame(self, path_frame: Optional[cv.typing.MatLike]) -> None:
        self._path_frame = path_frame.copy() if path_frame is not None else None
        
        if self._canvas is None:
            self._canvas = np.zeros_like(path_frame, np.uint8) if path_frame is not None else None


    @property
    def finishline_frame(self) -> Optional[cv.typing.MatLike]:
        return self._finishline_frame


    @finishline_frame.setter
    def finishline_frame(self, finishline_frame: Optional[cv.typing.MatLike]) -> None:
        self._finishline_frame = finishline_frame.copy() if finishline_frame is not None else None


    @property
    def color(self) -> SpheroColor:
        return self._color

    @color.setter
    def color(self, color: SpheroColor) -> None:
        self._color = color

    @property
    def username(self) -> str:
        return self._username

    @username.setter
    def username(self, username: str) -> None:
        self._username = username


    @property
    def background(self) -> Optional[cv.typing.MatLike]:
        return self._background

    @background.setter
    def background(self, background: Optional[cv.typing.MatLike]) -> None:
        self._background = background.copy() if background is not None else None


    @property
    def is_started(self) -> bool:
        return self._is_started

    @is_started.setter
    def is_started(self, is_started: bool) -> None:
        self._is_started = is_started


    @property
    def is_finished(self) -> bool:
        return self._is_finished

    @is_finished.setter
    def is_finished(self, is_finished: bool) -> None:
        self._is_finished = is_finished


    @property
    def is_lap_started(self) -> bool:
        return self._is_lap_started

    @is_lap_started.setter
    def is_lap_started(self, is_lap_started: bool) -> None:
        self._is_lap_started = is_lap_started


    @property
    def is_lap_stopped(self) -> bool:
        return self._is_lap_stopped

    @is_lap_stopped.setter
    def is_lap_stopped(self, is_lap_stopped: bool) -> None:
        self._is_lap_stopped = is_lap_stopped



    @property
    def start_time(self) -> Optional[datetime]:
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: Optional[datetime]) -> None:
        if self._is_started:
            self._start_time = start_time

    @property
    def finish_time(self) -> Optional[datetime]:
        return self._finish_time

    @finish_time.setter
    def finish_time(self, finish_time: Optional[datetime]) -> None:
        if self._is_finished:
            self._finish_time = finish_time

    @property
    def total_lap_time(self) -> Optional[float]:
        return self._total_lap_time

    @total_lap_time.setter
    def total_lap_time(self, total_lap_time: Optional[float]) -> None:
        if self._is_finished:
            self._total_lap_time = total_lap_time


    @property
    def path_previous_center(self) -> Optional[Tuple[int, int]]:
        return self._path_previous_center

    @path_previous_center.setter
    def path_previous_center(self, path_previous_center: Optional[Tuple[int, int]]) -> None:
        self._path_previous_center = path_previous_center

    @property
    def path_center(self) -> Optional[Tuple[int, int]]:
        return self._path_center

    @path_center.setter
    def path_center(self, path_center: Optional[Tuple[int, int]]) -> None:
        self._path_center = path_center

    @property
    def path_radius(self) -> Optional[int]:
        return self._path_radius

    @path_radius.setter
    def path_radius(self, path_radius: Optional[int]) -> None:
        self._path_radius = path_radius



    @property
    def finishline_center(self) -> Optional[Tuple[int, int]]: 
        return self._finishline_center

    @finishline_center.setter
    def finishline_center(self, finishline_center: Optional[Tuple[int, int]]) -> None:
        self._finishline_center = finishline_center


    @property
    def finishline_previous_center(self) -> Optional[Tuple[int, int]]: 
        return self._finishline_previous_center

    @finishline_previous_center.setter
    def finishline_previous_center(self, finishline_previous_center: Optional[Tuple[int, int]]) -> None:
        self._finishline_previous_center = finishline_previous_center



    @property
    def finishline_radius(self) -> Optional[int]:
        return self._finishline_radius

    @finishline_radius.setter
    def finishline_radius(self, finishline_radius: Optional[int]) -> None:
        self._finishline_radius = finishline_radius


    @property
    def canvas(self) -> Optional[cv.typing.MatLike]:
        return self._canvas

    @canvas.setter
    def canvas(self, frame: Optional[cv.typing.MatLike]) -> None:
        self._canvas = np.zeros_like(frame, np.uint8) if frame is not None else None

    
    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, debug: bool) -> None:
        self._debug = debug
   

    def get_processed_path_frame(
            self, 
            hsv_ranges: HsvColorsRange = HsvColorsRange.NORMAL, 
            min_radius: Optional[int] = None, 
            max_radius: Optional[int] = None, 
            bilateral_diameter: int = 9,
            bilateral_sigma_color: int = 75,
            bilateral_sigma_space: int = 75,
            median_kernel_size: int = 9,
            clahe_clip_limit : float = 4.0,
            clahe_tile_grid_size : int = 9,
            morph_kernel_size: int = 5,
            morph_iterator: int = 1,
            contours_chain_approx_simple: bool = True
            ) -> Optional[cv.typing.MatLike]:
        
        from backend.detectors.detector import Detector

        return Detector.get_detected_path_frame(
            sphero_bolt=self,
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple, 
            debug=self._debug
            )




    def get_processed_finishline_frame(
            self, 
            hsv_ranges: HsvColorsRange = HsvColorsRange.NORMAL, 
            min_radius: Optional[int] = None, 
            max_radius: Optional[int] = None,
            start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
            finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
            bilateral_diameter: int = 9,
            bilateral_sigma_color: int = 75,
            bilateral_sigma_space: int = 75,
            median_kernel_size: int = 9,
            clahe_clip_limit : float = 4.0,
            clahe_tile_grid_size : int = 9,
            morph_kernel_size: int = 5,
            morph_iterator: int = 1,
            contours_chain_approx_simple: bool = True
            ) -> Optional[cv.typing.MatLike]:
        
        from backend.detectors.detector import Detector


        return Detector.get_detected_finishline_frame(
            sphero_bolt=self, 
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius,
            start_line=start_line, 
            finish_line=finish_line, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple,
            debug=self._debug
            )





