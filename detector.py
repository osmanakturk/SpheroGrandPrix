import uuid
import time
import numpy as np
import math
import cv2 as cv
from datetime import datetime
from config import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR
from sphero_bolt import SpheroBold



class Detector:
        
    def __init__(self, 
                detector_type: str, 
                frame: cv.typing.MatLike = None, 
                sphero_bolt: SpheroBold = None, 
                path_min_radius: int = None, 
                path_max_radius: int = None, 
                finishline_min_radius: int = None, 
                finishline_max_radius: int = None, 
                debug: bool = False,
                kernel_size: int = 9 
                ):
            

        
        self._detector_type = detector_type
        self._frame = frame
        self._sphero_bolt = sphero_bolt
        self._path_min_radius = path_min_radius
        self._path_max_radius = path_max_radius
        self._finishline_min_radius = finishline_min_radius
        self._finishline_max_radius = finishline_max_radius
        self._debug = debug
        self._kernel_size = kernel_size


        
    
    @property
    def detector_type(self):
        return self._detector_type
    
    @detector_type.setter
    def detector_type(self, detector_type: str):
        self._detector_type = detector_type


    @property
    def frame(self):
        return self._frame
    
    @frame.setter
    def frame(self, frame: cv.typing.MatLike):
        self._frame = frame


    @property
    def sphero_bolt(self):
        return self._sphero_bolt
    
    @sphero_bolt.setter
    def sphero_bolt(self, sphero_bolt: SpheroBold):
        self._sphero_bolt = sphero_bolt


    @property
    def path_min_radius(self):
        return self._path_min_radius
    
    @path_min_radius.setter
    def path_min_radius(self, path_min_radius: int):
        self._path_min_radius = path_min_radius

    
    @property
    def path_max_radius(self):
        return self._path_max_radius

    @path_max_radius.setter
    def path_max_radius(self, path_max_radius: int):
        self._path_max_radius = path_max_radius
   
   
    
    @property
    def finishline_min_radius(self):
        return self._finishline_min_radius

    @finishline_min_radius.setter
    def finishline_min_radius(self, finishline_min_radius: int):
        self._finishline_min_radius = finishline_min_radius
   
   
    
    @property
    def finishline_max_radius(self):
        return self._finishline_max_radius

    @finishline_max_radius.setter
    def finishline_max_radius(self, finishline_max_radius: int):
        self._finishline_max_radius = finishline_max_radius
   

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: int):
        self._kernel_size = kernel_size
   

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, debug: int):
        self._debug = debug
   


   
    



    def get_processed_frame(self) -> cv.typing.MatLike:

        original_frame = self.frame.copy()
        median_blur = cv.medianBlur(self._frame.copy(), self._kernel_size)
        cv.imshow("median_blur", median_blur)
        #self._gaussian_blur = cv.GaussianBlur(self._frame.copy(), (self._kernel_size, self._kernel_size), 1.4)
        self._hsv = cv.cvtColor(self._frame.copy(), cv.COLOR_BGR2HSV)
        self._hsv[:, :, 2] = cv.equalizeHist(self._hsv[:, :, 2])
        
        match(self._sphero_bolt._color):

            case "Red":
                red1_mask = cv.inRange(self._hsv, COLOR_RANGES["Red1"]["Lower"], COLOR_RANGES["Red1"]["Upper"])
                red2_mask = cv.inRange(self._hsv, COLOR_RANGES["Red2"]["Lower"], COLOR_RANGES["Red2"]["Upper"])
                self._mask = cv.bitwise_or(red1_mask, red2_mask)

            case "Yellow":
                self._mask = cv.inRange(self._hsv, COLOR_RANGES["Yellow"]["Lower"], COLOR_RANGES["Yellow"]["Upper"])
                
            case "Green":
                self._mask = cv.inRange(self._hsv, COLOR_RANGES["Green"]["Lower"], COLOR_RANGES["Green"]["Upper"])
                
            case "Blue":
                self._mask = cv.inRange(self._hsv, COLOR_RANGES["Blue"]["Lower"], COLOR_RANGES["Blue"]["Upper"])
                

            case "Purple":
                self._mask = cv.inRange(self._hsv, COLOR_RANGES["Purple"]["Lower"], COLOR_RANGES["Purple"]["Upper"])



        
        masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
        cv.imshow("masked frame", masked_frame)

        masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
        cv.imshow("masked_gray_frame", masked_gray_frame)

        _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)
        cv.imshow("masked_frame_threshold", masked_frame_threshold)
        
       


        contours, hierarchy = cv.findContours(masked_frame_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        

        for contour in contours:
            area = cv.contourArea(contour)

            (x, y), radius = cv.minEnclosingCircle(contour)

            x = int(x)
            y = int(y)
            radius = int(radius)

            

            #self._x, self._y, self._w, self._h = cv.boundingRect(contour)
            #self._radius = int((self._w + self._h)//4)
            #self._x = int((self._x + self._w//2))
            #self._y = int((self._y + self._h//2))
                

            if self._detector_type == "Finishline":

                if radius >= self._finishline_min_radius and radius <= self._finishline_max_radius:
                    self._sphero_bolt.finishline_centre = (x, y)
                    self._sphero_bolt.finishline_radius = radius

                    cv.circle(original_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], -1, cv.LINE_AA)
                    
                    cv.putText(original_frame, self._sphero_bolt.username, (x, y-2*radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                    

            if self._detector_type == "Path":

                if radius >= self._path_min_radius and radius <= self._path_max_radius:
                    
                    self._sphero_bolt.path_centre = (x, y)
                    self._sphero_bolt.path_radius = radius

                    if self._sphero_bolt.path_previous_centre is None:
                        self._sphero_bolt.path_previous_centre = (x, y)
                    
                    cv.circle(original_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                    
                    #cv.circle(original_frame, (x, y), radius+2, COLORS_BGR["Black"], 2, cv.LINE_AA)
                    
                    cv.line(original_frame, self._sphero_bolt.path_previous_centre, 
                            (x, y),  COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                    
                    cv.line(self._sphero_bolt.canvas, self._sphero_bolt.path_previous_centre, 
                            (x, y), COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                    
                    self._sphero_bolt.path_previous_centre = (x, y)

                    cv.putText(original_frame, self._sphero_bolt.username, (x, y-2*radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                    
                    print(f"area: {area}, x: {x}, y: {y}, radius: {radius}")
                        

        return cv.bitwise_or(original_frame, self._sphero_bolt.canvas)



