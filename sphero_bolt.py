import uuid
import time
import numpy as np
import cv2 as cv
from datetime import datetime
from config import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR




class SpheroBold():

    def __init__(self, 
                 color: str, 
                 frame: cv.typing.MatLike, 
                 background:cv.typing.MatLike, 
                 username: str = "User", 
                 is_active: bool = False, 
                 start_time: datetime = None, 
                 finish_time: datetime = None, 
                 path_previous_points: tuple = None, 
                 path_center: tuple = None, 
                 path_radius: int = None,
                 finishline_center: tuple = None,
                 finishline_radius: int = None,

                 ):
        

        self._color = color
        self._username = username
        self._background = background
        self._is_active = is_active
        self._start_time = start_time
        self._finish_time = finish_time
        self._path_previous_points = path_previous_points
        self._path_center = path_center
        self._path_radius = path_radius
        self._finishline_center = finishline_center
        self._finishline_radius = finishline_radius
        self._canvas = np.full_like(frame, 0, np.uint8)

    

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
    def path_previous_points(self):
        return self._path_previous_points

    @path_previous_points.setter
    def path_previous_points(self, path_previous_points: tuple):
        self._path_previous_points = path_previous_points

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
    def finishline_radius(self, frame: cv.typing.MatLike):
        self._finishline_radius = np.full_like(frame, 0, np.uint8)


