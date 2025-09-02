import uuid
import time
import numpy as np
import cv2 as cv
from datetime import datetime
from backend.constants import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR



class SpheroBold():

    def __init__(self, 
                 color: str, 
                 username: str = "User",
                 path_frame: cv.typing.MatLike = None, 
                 finishline_frame: cv.typing.MatLike = None,
                 background: cv.typing.MatLike = None, 
                 is_active: bool = False, 
                 start_time: datetime = None, 
                 finish_time: datetime = None, 
                 path_previous_centre: tuple = None, 
                 path_centre: tuple = None, 
                 path_radius: int = None,
                 finishline_centre: tuple = None,
                 finishline_radius: int = None,
                 debug: bool = False
                 ):
        

        self._id = uuid.uuid4().hex
        self._color = color
        self._username = username
        self._background = background
        self._is_active = is_active
        self._start_time = start_time
        self._finish_time = finish_time
        self._path_previous_centre = path_previous_centre
        self._path_centre = path_centre
        self._path_radius = path_radius
        self._finishline_centre = finishline_centre
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
    def color(self, color: str):
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
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, is_active: bool):
        self._is_active = is_active

    @property
    def start_time(self):
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: datetime):
        self._start_time = start_time

    @property
    def finish_time(self):
        return self._finish_time

    @finish_time.setter
    def finish_time(self, finish_time: datetime):
        self._finish_time = finish_time

    @property
    def path_previous_centre(self):
        return self._path_previous_centre

    @path_previous_centre.setter
    def path_previous_centre(self, path_previous_centre: tuple):
        self._path_previous_centre = path_previous_centre

    @property
    def path_centre(self):
        return self._path_centre

    @path_centre.setter
    def path_centre(self, path_centre: tuple):
        self._path_centre = path_centre

    @property
    def path_radius(self):
        return self._path_radius

    @path_radius.setter
    def path_radius(self, path_radius: int):
        self._path_radius = path_radius



    @property
    def finishline_centre(self): 
        return self._finishline_centre

    @finishline_centre.setter
    def finishline_centre(self, finishline_centre: tuple):
        self._finishline_centre = finishline_centre



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
   

