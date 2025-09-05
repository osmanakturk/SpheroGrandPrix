import uuid
import time
import numpy as np
import math
import cv2 as cv
from typing import Optional, Tuple
from datetime import datetime
from backend.constants import COLOR_RANGES_STRICT, COLOR_RANGES_WIDE, COLORS_HSV, COLORS_BGR, COLOR_RANGES_NORMAL, COLOR_RANGES_MANUAL
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
                color_type: str = "Normal",
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
        self._color_type = color_type
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
    def color_type(self):
        return self._color_type

    @color_type.setter
    def color_type(self, color_type: str):
        self._color_type = color_type


        
    @staticmethod
    def get_detected_path_frame(
        frame: cv.typing.MatLike, 
        color: str, 
        canvas: cv.typing.MatLike,
        username: Optional[str] = None,
        color_type: str = "Normal", 
        min_radius: Optional[int] = None, 
        max_radius: Optional[int] = None,
        centre: Optional[Tuple[int, int]] = None,
        prev_centre: Optional[Tuple[int, int]] = None,
        kernel_size: int = 9,
        debug: bool = False
        ):

        pass
        






    def get_processed_path_frame(self) -> cv.typing.MatLike:

        original_path_frame = self.path_frame.copy()
        

        median_blur = cv.medianBlur(self._finishline_frame.copy(), self._kernel_size)

        if self._debug:
            cv.imshow("median_blur", median_blur)
        
        #self._gaussian_blur = cv.GaussianBlur(self._frame.copy(), (self._kernel_size, self._kernel_size), 1.4)
        self._hsv = cv.cvtColor(self._finishline_frame.copy(), cv.COLOR_BGR2HSV)
        self._hsv[:, :, 2] = cv.equalizeHist(self._hsv[:, :, 2])
        

        match(self._color_type):
            case "Normal":
                COLORS = COLOR_RANGES_NORMAL
            case "Wide":
                COLORS = COLOR_RANGES_WIDE
            case "Strict":
                COLORS = COLOR_RANGES_STRICT
            case "Manual":
                COLORS = COLOR_RANGES_MANUAL




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
                
                self._sphero_bolt.path_center = (x, y)
                self._sphero_bolt.path_radius = radius
                if self._sphero_bolt.path_previous_center is None:
                    self._sphero_bolt.path_previous_center = (x, y)
                
                cv.circle(original_path_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                
                #cv.circle(original_frame, (x, y), radius+2, COLORS_BGR["Black"], 2, cv.LINE_AA)
                
                cv.line(original_path_frame, self._sphero_bolt.path_previous_center, 
                        (x, y),  COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                
                cv.line(self._sphero_bolt.canvas, self._sphero_bolt.path_previous_center, 
                        (x, y), COLORS_BGR[self._sphero_bolt.color], 3, cv.LINE_AA)
                
                self._sphero_bolt.path_previous_center = (x, y)

                cv.putText(original_path_frame, self._sphero_bolt.username, (x, y-2*radius), 
                        cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                
                if self._debug:
                    print(f"Path, area: {area}, x: {x}, y: {y}, radius: {radius}")
                        


        return cv.bitwise_or(original_path_frame, self._sphero_bolt.canvas)



    @staticmethod
    def get_detected_finishline_frame(
        frame: cv.typing.MatLike, 
        sphero_color: str, 
        sphero_username: str,
        color_type: str = "Wide", 
        min_radius: Optional[int] = None, 
        max_radius: Optional[int] = None,
        sphero_finishline_center: Optional[Tuple[int, int]] = None,
        sphero_finishline_radius: Optional[int] = None,
        start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
        stop_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
        bilateral_diameter: int = 25,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        median_kernel_size: int = 25,
        clahe_clip_limit : float = 4.0,
        clahe_tile_grid_size : int = 9,
        morph_kernel_size: int = 5,
        morph_iterator: int = 1,
        contours_chain_approx_simple: bool = True,
        debug: bool = False
        ) -> Tuple[cv.typing.MatLike, Optional[Tuple[int, int]], Optional[int]]:
            
            bilateral_diameter = max(1, bilateral_diameter if bilateral_diameter%2 == 1 else bilateral_diameter+1)
            bilateral_sigma_color = max(1, bilateral_sigma_color)
            bilateral_sigma_space = max(1, bilateral_sigma_space)
            median_kernel_size = max(1, median_kernel_size if median_kernel_size%2 == 1 else median_kernel_size + 1)
            morph_kernel_size = max(1, morph_kernel_size if morph_kernel_size%2==1 else morph_kernel_size+1)
            morph_iterator = max(1, morph_iterator)
            clahe_clip_limit = max(1.0, clahe_clip_limit)
            clahe_tile_grid_size = max(2, clahe_tile_grid_size)




            original_frame_copy = frame.copy()

            finishline_bilateral = cv.bilateralFilter(frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if debug:
                cv.imshow("Finishline Bilateral", finishline_bilateral)

       

            hsv_bilateral = cv.cvtColor(finishline_bilateral, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv_bilateral)
            clahe = cv.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            v_clahe = clahe.apply(v)
            
            hsv = cv.merge([h, s, v_clahe])

                    
            match(color_type):
                case "Normal":
                    COLORS = COLOR_RANGES_NORMAL
                case "Wide":
                    COLORS = COLOR_RANGES_WIDE
                case "Strict":
                    COLORS = COLOR_RANGES_STRICT
                case "Manual":
                    COLORS = COLOR_RANGES_MANUAL
            

            contours = None

            if sphero_color == "Red":
                red1_mask = cv.inRange(hsv, COLORS["Red1"]["Lower"], COLORS["Red1"]["Upper"])
                red2_mask = cv.inRange(hsv, COLORS["Red2"]["Lower"], COLORS["Red2"]["Upper"])
                mask = cv.bitwise_or(red1_mask, red2_mask)
            else:
                mask = cv.inRange(hsv, COLORS[sphero_color]["Lower"], COLORS[sphero_color]["Upper"])

          

            median_mask = cv.medianBlur(mask, median_kernel_size)
            morph_ellipse_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

            median_mask_morph = cv.morphologyEx(median_mask, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            median_mask_morph = cv.morphologyEx(median_mask_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)

            masked_frame = cv.bitwise_and(frame, frame, mask=median_mask_morph)

            if contours_chain_approx_simple:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            if debug:
                cv.imshow(f"{sphero_color} Masked", masked_frame)
                cv.imshow(f"{sphero_color} Median Mask", median_mask)
                cv.imshow(f"{sphero_color} Median Mask Morphology", median_mask_morph)
                contours_frame = cv.drawContours(frame.copy(), contours, -1, (0, 0, 255), 3)
                cv.imshow(f"{sphero_color} Contours", contours_frame)
                    

            

            if contours:
                for contour in contours:
                    area = cv.contourArea(contour)

                    (x, y), radius = cv.minEnclosingCircle(contour)

                    x = int(x)
                    y = int(y)
                    radius = int(radius)

                    #x, y, w, h = cv.boundingRect(contour)
                    #radius = int((w + h)//4)
                    #x = int((x + w//2))
                    #y = int((y + h//2))
                        
                        
                    if min_radius is not None and max_radius is not None and radius >= min_radius and radius <= max_radius:
                        
                        sphero_finishline_center = (x, y)
                        sphero_finishline_radius = radius

                        cv.circle(original_frame_copy, (x, y), radius, COLORS_BGR[sphero_color], 2, cv.LINE_AA)
                        
                        cv.putText(original_frame_copy, sphero_username, (x, y-2*radius), 
                                cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_color], 2, cv.LINE_AA)
                            

                    if debug:
                        print(f"Finishline, area: {area}, x: {x}, y: {y}, radius: {radius}")




            if start_line is not None:
                cv.line(original_frame_copy, start_line[0], start_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)


            if stop_line is not None:
                cv.line(original_frame_copy, stop_line[0], stop_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)


            return (original_frame_copy, sphero_finishline_center, sphero_finishline_radius)



    def get_processed_finishline_frame(self) -> cv.typing.MatLike:


            original_finishline_frame = self.finishline_frame.copy()

            median_blur = cv.medianBlur(self._finishline_frame.copy(), self._kernel_size)

            if self._debug:
                cv.imshow("median_blur", median_blur)

            #self._gaussian_blur = cv.GaussianBlur(self._frame.copy(), (self._kernel_size, self._kernel_size), 1.4)
            self._hsv = cv.cvtColor(self._finishline_frame.copy(), cv.COLOR_BGR2HSV)
            self._hsv[:, :, 2] = cv.equalizeHist(self._hsv[:, :, 2])

                    
            match(self._color_type):
                case "Normal":
                    COLORS = COLOR_RANGES_NORMAL
                case "Wide":
                    COLORS = COLOR_RANGES_WIDE
                case "Strict":
                    COLORS = COLOR_RANGES_STRICT
                case "Manual":
                    COLORS = COLOR_RANGES_MANUAL
            
            
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
                    
                    self._sphero_bolt.finishline_center = (x, y)
                    self._sphero_bolt.finishline_radius = radius
                    cv.circle(original_finishline_frame, (x, y), radius, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                    
                    cv.putText(original_finishline_frame, self._sphero_bolt.username, (x, y-2*radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[self._sphero_bolt.color], 2, cv.LINE_AA)
                        
                if self._debug:
                    print(f"Finishline, area: {area}, x: {x}, y: {y}, radius: {radius}")


            cv.line(original_finishline_frame, (0, self._finishline_y), (original_finishline_frame.shape[1], self._finishline_y), COLORS_BGR["Red"], 2, cv.LINE_AA)

            return original_finishline_frame




