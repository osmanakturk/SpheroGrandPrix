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
                 color:  SpheroColor, 
                 username: Optional[str] = None,
                 path_frame: Optional[cv.typing.MatLike] = None, 
                 finishline_frame: Optional[cv.typing.MatLike] = None,
                 background: Optional[cv.typing.MatLike] = None, 
                 is_started: bool = False,
                 is_finished: bool = False,
                 start_time: Optional[datetime] = None, 
                 finish_time: Optional[datetime] = None, 
                 lap_time: Optional[datetime] = None,
                 path_previous_center: Optional[Tuple[int, int]] = None, 
                 path_center: Optional[Tuple[int, int]] = None, 
                 path_radius: Optional[int] = None,
                 finishline_center: Optional[Tuple[int, int]] = None,
                 finishline_radius: Optional[int] = None,
                 debug: bool = False
                 ):
        

        self._id = uuid.uuid4().hex
        self._color = color
        self._username = username if username is not None else self._color
        self._background = background
        self._is_started = is_started
        self._is_finished = is_finished
        self._start_time = start_time
        self._finish_time = finish_time
        self._lap_time = lap_time
        self._path_previous_center = path_previous_center
        self._path_center = path_center
        self._path_radius = path_radius
        self._finishline_center = finishline_center
        self._finishline_radius = finishline_radius
        self._path_frame = path_frame
        self._finishline_frame = finishline_frame
        self._canvas = np.full_like(self._path_frame, 0, np.uint8)
        self._debug = debug
  




    @property
    def path_frame(self):
        return self._path_frame


    @path_frame.setter
    def path_frame(self, path_frame: cv.typing.MatLike):
        self._path_frame = path_frame


    @property
    def finishline_frame(self):
        return self.finishline_frame


    @finishline_frame.setter
    def finishline_frame(self, finishline_frame: cv.typing.MatLike):
        self._finishline_frame = finishline_frame


    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, color: SpheroColor):
        self._color = color

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, username: str):
        self._username = username


    @property
    def background(self):
        return self._background

    @background.setter
    def background(self, background: cv.typing.MatLike):
        self._background = background


    @property
    def is_started(self):
        return self._is_started

    @is_started.setter
    def is_started(self, is_started: bool):
        self._is_started = is_started


    @property
    def is_finished(self):
        return self._is_finished

    @is_finished.setter
    def is_finished(self, is_finished: bool):
        self._is_finished = is_finished


    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: datetime):
        if self._is_started:
            self._start_time = start_time

    @property
    def finish_time(self):
        return self._finish_time

    @finish_time.setter
    def finish_time(self, finish_time: datetime):
        if self._is_finished:
            self._finish_time = finish_time

    @property
    def lap_time(self):
        return self._lap_time

    @lap_time.setter
    def lap_time(self, lap_time: datetime):
        if self._is_finished:
            self._lap_time = lap_time


    @property
    def path_previous_center(self):
        return self._path_previous_center

    @path_previous_center.setter
    def path_previous_center(self, path_previous_center: tuple):
        self._path_previous_center = path_previous_center

    @property
    def path_center(self):
        return self._path_center

    @path_center.setter
    def path_center(self, path_center: tuple):
        self._path_center = path_center

    @property
    def path_radius(self):
        return self._path_radius

    @path_radius.setter
    def path_radius(self, path_radius: int):
        self._path_radius = path_radius



    @property
    def finishline_center(self): 
        return self._finishline_center

    @finishline_center.setter
    def finishline_center(self, finishline_center: tuple):
        self._finishline_center = finishline_center



    @property
    def finishline_radius(self):
        return self._finishline_radius

    @finishline_radius.setter
    def finishline_radius(self, finishline_radius: int):
        self._finishline_radius = finishline_radius


    @property
    def canvas(self):
        return self._canvas

    @canvas.setter
    def canvas(self, frame: cv.typing.MatLike):
        self._canvas = np.full_like(frame, 0, np.uint8)

    
    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, debug: bool):
        self._debug = debug
   

