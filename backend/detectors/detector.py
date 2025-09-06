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
        
        
    @staticmethod
    def get_detected_path_frame(
        frame: cv.typing.MatLike, 
        sphero_color: str, 
        sphero_canvas: cv.typing.MatLike,
        sphero_username: str,
        color_type: str = "Normal", 
        min_radius: Optional[int] = None, 
        max_radius: Optional[int] = None,
        sphero_path_center: Optional[Tuple[int, int]] = None,
        sphero_path_previous_center: Optional[Tuple[int, int]] = None,
        sphero_path_radius: Optional[int] = None,
        bilateral_diameter: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        median_kernel_size: int = 9,
        clahe_clip_limit : float = 4.0,
        clahe_tile_grid_size : int = 9,
        morph_kernel_size: int = 5,
        morph_iterator: int = 1,
        contours_chain_approx_simple: bool = True,
        debug: bool = False
        ) -> Tuple[Optional[cv.typing.MatLike], 
                   Optional[cv.typing.MatLike], 
                   Optional[Tuple[int, int]], 
                   Optional[Tuple[int, int]], 
                   Optional[int]]:


            bilateral_diameter = max(1, bilateral_diameter if bilateral_diameter%2 == 1 else bilateral_diameter+1)
            bilateral_sigma_color = max(1, bilateral_sigma_color)
            bilateral_sigma_space = max(1, bilateral_sigma_space)
            median_kernel_size = max(1, median_kernel_size if median_kernel_size%2 == 1 else median_kernel_size + 1)
            morph_kernel_size = max(1, morph_kernel_size if morph_kernel_size%2==1 else morph_kernel_size+1)
            morph_iterator = max(1, morph_iterator)
            clahe_clip_limit = max(1.0, clahe_clip_limit)
            clahe_tile_grid_size = max(2, clahe_tile_grid_size)


            original_frame_copy = frame.copy()

            path_bilateral = cv.bilateralFilter(frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if debug:
                cv.imshow("Path Bilateral", path_bilateral)

       

            hsv_bilateral = cv.cvtColor(path_bilateral, cv.COLOR_BGR2HSV)
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
                        
                        sphero_path_center = (x, y)
                        sphero_path_radius = radius

                        if sphero_path_previous_center is None:
                            sphero_path_previous_center = (x, y)

                        cv.circle(original_frame_copy, sphero_path_center, radius, 
                                  COLORS_BGR[sphero_color], 2, cv.LINE_AA)

                        cv.line(original_frame_copy, sphero_path_previous_center, sphero_path_center, 
                                COLORS_BGR[sphero_color], 3, cv.LINE_AA)
                
                        cv.line(sphero_canvas, sphero_path_previous_center, sphero_path_center, 
                                COLORS_BGR[sphero_color], 3, cv.LINE_AA)

                        sphero_path_previous_center = sphero_path_center


                        cv.putText(original_frame_copy, sphero_username, (x, y-2*radius), 
                                cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_color], 2, cv.LINE_AA)
                            

                    if debug:
                        print(f"Path, area: {area}, x: {x}, y: {y}, radius: {radius}")




    
            return (original_frame_copy, sphero_canvas, sphero_path_center, sphero_path_previous_center, sphero_path_radius)
        




    @staticmethod
    def get_detected_finishline_frame(
        frame: cv.typing.MatLike, 
        sphero_color: str, 
        sphero_username: str,
        color_type: str = "Normal", 
        min_radius: Optional[int] = None, 
        max_radius: Optional[int] = None,
        sphero_finishline_center: Optional[Tuple[int, int]] = None,
        sphero_finishline_radius: Optional[int] = None,
        start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
        stop_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
        bilateral_diameter: int = 9,
        bilateral_sigma_color: int = 75,
        bilateral_sigma_space: int = 75,
        median_kernel_size: int = 9,
        clahe_clip_limit : float = 4.0,
        clahe_tile_grid_size : int = 9,
        morph_kernel_size: int = 5,
        morph_iterator: int = 1,
        contours_chain_approx_simple: bool = True,
        debug: bool = False
        ) -> Tuple[Optional[cv.typing.MatLike], 
                   Optional[Tuple[int, int]], 
                   Optional[int]]:
            
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
    





