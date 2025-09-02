import uuid
import time
import numpy as np
import math
import cv2 as cv
from datetime import datetime
from backend.constants import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR
from backend.models.sphero_bolt import SpheroBold



class Detector:
        
    def __init__(self, 
                path_frame: cv.typing.MatLike = None, 
                finishline_frame: cv.typing.MatLike = None,
                sphero_bolt: SpheroBold = None, 
                path_min_radius: int = None, 
                path_max_radius: int = None, 
                finishline_min_radius: int = None, 
                finishline_max_radius: int = None, 
                finishline_y: int = None,
                debug: bool = False,
                is_strict: bool = True,
                kernel_size: int = 9 
                ):
            

        

        self._path_frame = path_frame
        self._finishline_frame = finishline_frame
        self._sphero_bolt = sphero_bolt
        self._path_min_radius = path_min_radius
        self._path_max_radius = path_max_radius
        self._finishline_min_radius = finishline_min_radius
        self._finishline_max_radius = finishline_max_radius
        self._finishline_y = finishline_y
        self._debug = debug
        self._is_strict = is_strict
        self._kernel_size = kernel_size


        


    @property
    def path_frame(self):
        return self._path_frame
    
    @path_frame.setter
    def path_frame(self, path_frame: cv.typing.MatLike):
        self._path_frame = path_frame



    @property
    def finishline_frame(self):
        return self._finishline_frame
    
    @finishline_frame.setter
    def finishline_frame(self, finishline_frame: cv.typing.MatLike):
        self._finishline_frame = finishline_frame



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
    def finishline_y(self):
        return self._finishline_y

    @finishline_y.setter
    def finishline_y(self, finishline_y: int):
        self._finishline_y = finishline_y


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
    def debug(self, debug: bool):
        self._debug = debug
   

    @property
    def is_strict(self):
        return self._is_strict

    @is_strict.setter
    def is_strict(self, is_strict: bool):
        self._is_strict = is_strict




    def get_processed_path_frame(self) -> cv.typing.MatLike:

        original_path_frame = self.path_frame.copy()
        

        median_blur = cv.medianBlur(self._finishline_frame.copy(), self._kernel_size)

        if self._debug:
            cv.imshow("median_blur", median_blur)
        
        #self._gaussian_blur = cv.GaussianBlur(self._frame.copy(), (self._kernel_size, self._kernel_size), 1.4)
        self._hsv = cv.cvtColor(self._finishline_frame.copy(), cv.COLOR_BGR2HSV)
        self._hsv[:, :, 2] = cv.equalizeHist(self._hsv[:, :, 2])
        
        if self._is_strict:
            COLORS = COLOR_RANGES_STRICT
        else:
            COLORS = COLOR_RANGES


        match(self._sphero_bolt._color):

            case "Red":
                red1_mask = cv.inRange(self._hsv, COLORS["Red1"]["Lower"], COLORS["Red1"]["Upper"])
                red2_mask = cv.inRange(self._hsv, COLORS["Red2"]["Lower"], COLORS["Red2"]["Upper"])
                self._mask = cv.bitwise_or(red1_mask, red2_mask)

                masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                if self._debug:
                    cv.imshow("Red masked_frame", masked_frame)
                    cv.imshow("Red masked_gray_frame", masked_gray_frame)
                    cv.imshow("Red masked_frame_threshold", masked_frame_threshold)
        

            case "Yellow":
                self._mask = cv.inRange(self._hsv, COLORS["Yellow"]["Lower"], COLORS["Yellow"]["Upper"])

                masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)


                if self._debug:
                    cv.imshow("Yellow masked_frame", masked_frame)
                    cv.imshow("Yellow masked_gray_frame", masked_gray_frame)
                    cv.imshow("Yellow masked_frame_threshold", masked_frame_threshold)
        
        

                
            case "Green":
                self._mask = cv.inRange(self._hsv, COLORS["Green"]["Lower"], COLORS["Green"]["Upper"])
                
                masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                
                if self._debug:
                    cv.imshow("Green masked_frame", masked_frame)
                    cv.imshow("Green masked_gray_frame", masked_gray_frame)
                    cv.imshow("Green masked_frame_threshold", masked_frame_threshold)
        

                
            case "Blue":
                self._mask = cv.inRange(self._hsv, COLORS["Blue"]["Lower"], COLORS["Blue"]["Upper"])
                
                masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)
        
                if self._debug:
                    cv.imshow("Blue masked_frame", masked_frame)
                    cv.imshow("Blue masked_gray_frame", masked_gray_frame)
                    cv.imshow("Blue masked_frame_threshold", masked_frame_threshold)
        

                

            case "Purple":
                self._mask = cv.inRange(self._hsv, COLORS["Purple"]["Lower"], COLORS["Purple"]["Upper"])

                masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                if self._debug:
                    cv.imshow("Purple masked_frame", masked_frame)
                    cv.imshow("Purple masked_gray_frame", masked_gray_frame)
                    cv.imshow("Purple masked_frame_threshold", masked_frame_threshold)
        
        
       
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
                

            if radius >= self._path_min_radius and radius <= self._path_max_radius:
                
                self._sphero_bolt.path_centre = (x, y)
                self._sphero_bolt.path_radius = radius
                if self._sphero_bolt.path_previous_centre is None:
                    self._sphero_bolt.path_previous_centre = (x, y)
                
                cv.circle(original_path_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                
                #cv.circle(original_frame, (x, y), radius+2, COLORS_BGR["Black"], 2, cv.LINE_AA)
                
                cv.line(original_path_frame, self._sphero_bolt.path_previous_centre, 
                        (x, y),  COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                
                cv.line(self._sphero_bolt.canvas, self._sphero_bolt.path_previous_centre, 
                        (x, y), COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                
                self._sphero_bolt.path_previous_centre = (x, y)

                cv.putText(original_path_frame, self._sphero_bolt.username, (x, y-2*radius), 
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                
                if self._debug:
                    print(f"Path, area: {area}, x: {x}, y: {y}, radius: {radius}")
                        


        return cv.bitwise_or(original_path_frame, self._sphero_bolt.canvas)






    def get_processed_finishline_frame(self) -> cv.typing.MatLike:


            original_finishline_frame = self.finishline_frame.copy()

            median_blur = cv.medianBlur(self._finishline_frame.copy(), self._kernel_size)

            if self._debug:
                cv.imshow("median_blur", median_blur)

            #self._gaussian_blur = cv.GaussianBlur(self._frame.copy(), (self._kernel_size, self._kernel_size), 1.4)
            self._hsv = cv.cvtColor(self._finishline_frame.copy(), cv.COLOR_BGR2HSV)
            self._hsv[:, :, 2] = cv.equalizeHist(self._hsv[:, :, 2])

                    
            if self._is_strict:
                COLORS = COLOR_RANGES_STRICT
            else:
                COLORS = COLOR_RANGES

            
            match(self._sphero_bolt._color):

                case "Red":
                    red1_mask = cv.inRange(self._hsv, COLORS["Red1"]["Lower"], COLORS["Red1"]["Upper"])
                    red2_mask = cv.inRange(self._hsv, COLORS["Red2"]["Lower"], COLORS["Red2"]["Upper"])
                    self._mask = cv.bitwise_or(red1_mask, red2_mask)
                    
                    masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                    masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                    _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                    if self._debug:
                        cv.imshow("Red masked frame", masked_frame)
                        cv.imshow("Red masked_gray_frame", masked_gray_frame)
                        cv.imshow("Red masked_frame_threshold", masked_frame_threshold)
                    


                case "Yellow":
                    self._mask = cv.inRange(self._hsv, COLORS["Yellow"]["Lower"], COLORS["Yellow"]["Upper"])
           
                    masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                    masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                    _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                    if self._debug:
                        cv.imshow("Yellow masked frame", masked_frame)
                        cv.imshow("Yellow masked_gray_frame", masked_gray_frame)
                        cv.imshow("Yellow masked_frame_threshold", masked_frame_threshold)
            
                    
                case "Green":
                    self._mask = cv.inRange(self._hsv, COLORS["Green"]["Lower"], COLORS["Green"]["Upper"])
       
                    masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                    masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                    _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                    if self._debug:
                        cv.imshow("Green masked frame", masked_frame)
                        cv.imshow("Green masked_gray_frame", masked_gray_frame)
                        cv.imshow("Green masked_frame_threshold", masked_frame_threshold)
            
                    
                case "Blue":
                    self._mask = cv.inRange(self._hsv, COLORS["Blue"]["Lower"], COLORS["Blue"]["Upper"])
           
                    masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                    masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                    _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                    if self._debug:
                        cv.imshow("Blue masked frame", masked_frame)
                        cv.imshow("Blue masked_gray_frame", masked_gray_frame)
                        cv.imshow("Blue masked_frame_threshold", masked_frame_threshold)
            
                    

                case "Purple":
                    self._mask = cv.inRange(self._hsv, COLORS["Purple"]["Lower"], COLORS["Purple"]["Upper"])

                    masked_frame = cv.bitwise_and(median_blur.copy(), median_blur.copy(), mask=self._mask)
                    masked_gray_frame = cv.cvtColor(masked_frame.copy(), cv.COLOR_BGR2GRAY)
                    _, masked_frame_threshold= cv.threshold(masked_gray_frame.copy(), 50, 255, cv.THRESH_BINARY)

                    if self._debug:
                        cv.imshow("Purple masked frame", masked_frame)
                        cv.imshow("Purple masked_gray_frame", masked_gray_frame)
                        cv.imshow("Purple masked_frame_threshold", masked_frame_threshold)
            
            

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
                    


                    
                if radius >= self._finishline_min_radius and radius <= self._finishline_max_radius:
                    
                    self._sphero_bolt.finishline_centre = (x, y)
                    self._sphero_bolt.finishline_radius = radius
                    cv.circle(original_finishline_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                    
                    cv.putText(original_finishline_frame, self._sphero_bolt.username, (x, y-2*radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                        
                if self._debug:
                    print(f"Finishline, area: {area}, x: {x}, y: {y}, radius: {radius}")


            cv.line(original_finishline_frame, (0, self._finishline_y), (original_finishline_frame.shape[1], self._finishline_y), COLORS_BGR["Red"], 2, cv.LINE_AA)

            return original_finishline_frame




