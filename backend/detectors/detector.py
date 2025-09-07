import uuid
import time
import numpy as np
import math
import cv2 as cv
from typing import Optional, Tuple
from datetime import datetime
from backend.constants import HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_HSV, COLORS_BGR, HSV_RANGES_NORMAL, HSV_RANGES_MANUAL
from backend.models.sphero_bolt import SpheroBolt
from backend.utils import HsvColorsRange, SpheroColor



class Detector:
        
        
    @staticmethod
    def get_detected_path_frame(
        frame: cv.typing.MatLike, 
        sphero_bolt: SpheroBolt, 
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
        contours_chain_approx_simple: bool = True,
        debug: bool = False
        ) -> Optional[cv.typing.MatLike]:


            bilateral_diameter = max(1, bilateral_diameter if bilateral_diameter%2 == 1 else bilateral_diameter+1)
            bilateral_sigma_color = max(1, bilateral_sigma_color)
            bilateral_sigma_space = max(1, bilateral_sigma_space)
            median_kernel_size = max(1, median_kernel_size if median_kernel_size%2 == 1 else median_kernel_size + 1)
            morph_kernel_size = max(1, morph_kernel_size if morph_kernel_size%2==1 else morph_kernel_size+1)
            morph_iterator = max(1, morph_iterator)
            clahe_clip_limit = max(1.0, clahe_clip_limit)
            clahe_tile_grid_size = max(2, clahe_tile_grid_size)

            HSV_RANGES = hsv_ranges.value


            original_frame_copy = frame.copy()

            path_bilateral = cv.bilateralFilter(frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if debug:
                cv.imshow("Path Bilateral", path_bilateral)

       

            hsv_bilateral = cv.cvtColor(path_bilateral, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv_bilateral)
            clahe = cv.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            v_clahe = clahe.apply(v)
            
            hsv = cv.merge([h, s, v_clahe])

                    


            contours = None

            if sphero_bolt.color == SpheroColor.RED:
                red1_mask = cv.inRange(hsv, HSV_RANGES["Red1"]["Lower"], HSV_RANGES["Red1"]["Upper"])
                red2_mask = cv.inRange(hsv, HSV_RANGES["Red2"]["Lower"], HSV_RANGES["Red2"]["Upper"])
                mask = cv.bitwise_or(red1_mask, red2_mask)
            else:
                mask = cv.inRange(hsv, HSV_RANGES[sphero_bolt.color.value]["Lower"], HSV_RANGES[sphero_bolt.color.value]["Upper"])

          

            median_mask = cv.medianBlur(mask, median_kernel_size)

            morph_ellipse_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

            median_mask_morph = cv.morphologyEx(median_mask, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            median_mask_morph = cv.morphologyEx(median_mask_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)

            
            if contours_chain_approx_simple:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            if debug:
                masked_frame = cv.bitwise_and(frame, frame, mask=median_mask_morph)
                cv.imshow(f"{sphero_bolt.color.value} Masked", masked_frame)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask", median_mask)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask Morphology", median_mask_morph)
                contours_frame = cv.drawContours(frame.copy(), contours, -1, (0, 0, 255), 3)
                cv.imshow(f"{sphero_bolt.color.value} Contours", contours_frame)
            


            if contours:

                best_contour = None
                best_radius = -1
                best_contour_index = 0
                total_contours = len(contours)

                for idx, contour in enumerate(contours, 1):
                    
                    (x, y), radius = cv.minEnclosingCircle(contour)

                    x = int(x)
                    y = int(y)
                    radius = int(radius)
                   

                    #x, y, w, h = cv.boundingRect(contour)
                    #radius = int((w + h)//4)
                    #x = int((x + w//2))
                    #y = int((y + h//2))

                    if min_radius is not None and max_radius is not None and min_radius <= radius <= max_radius and radius > best_radius:
                        
                        best_contour = contour
                        best_radius = radius
                        best_contour_index = idx
                        best_area = cv.contourArea(contour)


                        
                if best_contour is not None and best_radius is not None:
                    
                    (b_x, b_y), b_radius = cv.minEnclosingCircle(best_contour)

                    b_x = int(b_x)
                    b_y = int(b_y)
                    b_radius = int(b_radius)

                    sphero_bolt.path_center = (b_x, b_y)
                    sphero_bolt.path_radius = b_radius

                    if sphero_bolt.path_previous_center is None:

                        sphero_bolt.path_previous_center = (b_x, b_y)

                    cv.circle(original_frame_copy, sphero_bolt.path_center, sphero_bolt.path_radius, 
                              COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                    
                    cv.line(original_frame_copy, sphero_bolt.path_previous_center, sphero_bolt.path_center, 
                            COLORS_BGR[sphero_bolt.color.value], 3, cv.LINE_AA)
            
                    cv.line(sphero_bolt.canvas, sphero_bolt.path_previous_center, sphero_bolt.path_center, 
                            COLORS_BGR[sphero_bolt.color.value], 3, cv.LINE_AA)
                    
                    sphero_bolt.path_previous_center = (b_x, b_y)

                    cv.putText(original_frame_copy, sphero_bolt.username, (b_x, b_y-2*sphero_bolt.path_radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                        
                    if debug:
                        print(f"Path, area: {best_area}, x: {b_x}, y: {b_y}, radius: {b_radius}, total contours: {total_contours}, best contour index: {best_contour_index}")



    
            return original_frame_copy
        




    @staticmethod
    def get_detected_finishline_frame(
        frame: cv.typing.MatLike, 
        sphero_bolt: SpheroBolt, 
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
        contours_chain_approx_simple: bool = True,
        debug: bool = False
        ) -> Optional[cv.typing.MatLike]:
            
            bilateral_diameter = max(1, bilateral_diameter if bilateral_diameter%2 == 1 else bilateral_diameter+1)
            bilateral_sigma_color = max(1, bilateral_sigma_color)
            bilateral_sigma_space = max(1, bilateral_sigma_space)
            median_kernel_size = max(1, median_kernel_size if median_kernel_size%2 == 1 else median_kernel_size + 1)
            morph_kernel_size = max(1, morph_kernel_size if morph_kernel_size%2==1 else morph_kernel_size+1)
            morph_iterator = max(1, morph_iterator)
            clahe_clip_limit = max(1.0, clahe_clip_limit)
            clahe_tile_grid_size = max(2, clahe_tile_grid_size)

            HSV_RANGES = hsv_ranges.value


            original_frame_copy = frame.copy()

            finishline_bilateral = cv.bilateralFilter(frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if debug:
                cv.imshow("Finishline Bilateral", finishline_bilateral)

       

            hsv_bilateral = cv.cvtColor(finishline_bilateral, cv.COLOR_BGR2HSV)
            h, s, v = cv.split(hsv_bilateral)
            clahe = cv.createCLAHE(clipLimit=float(clahe_clip_limit), tileGridSize=(clahe_tile_grid_size, clahe_tile_grid_size))
            v_clahe = clahe.apply(v)
            
            hsv = cv.merge([h, s, v_clahe])

                    

            contours = None

            if sphero_bolt.color == SpheroColor.RED:
                red1_mask = cv.inRange(hsv, HSV_RANGES["Red1"]["Lower"], HSV_RANGES["Red1"]["Upper"])
                red2_mask = cv.inRange(hsv, HSV_RANGES["Red2"]["Lower"], HSV_RANGES["Red2"]["Upper"])
                mask = cv.bitwise_or(red1_mask, red2_mask)
            else:
                mask = cv.inRange(hsv, HSV_RANGES[sphero_bolt.color.value]["Lower"], HSV_RANGES[sphero_bolt.color.value]["Upper"])

          

            median_mask = cv.medianBlur(mask, median_kernel_size)

            morph_ellipse_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

            median_mask_morph = cv.morphologyEx(median_mask, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            median_mask_morph = cv.morphologyEx(median_mask_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)


            if contours_chain_approx_simple:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            if debug:
                masked_frame = cv.bitwise_and(frame, frame, mask=median_mask_morph)
                cv.imshow(f"{sphero_bolt.color.value} Masked", masked_frame)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask", median_mask)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask Morphology", median_mask_morph)
                contours_frame = cv.drawContours(frame.copy(), contours, -1, (0, 0, 255), 3)
                cv.imshow(f"{sphero_bolt.color.value} Contours", contours_frame)
                    

            

            if contours:

                best_contour = None
                best_radius = -1
                best_contour_index = 0
                total_contours = len(contours)


                for idx, contour in enumerate(contours, 1):

                    (x, y), radius = cv.minEnclosingCircle(contour)

                    x = int(x)
                    y = int(y)
                    radius = int(radius)

                    

                    #x, y, w, h = cv.boundingRect(contour)
                    #radius = int((w + h)//4)
                    #x = int((x + w//2))
                    #y = int((y + h//2))
                        
                    if min_radius is not None and max_radius is not None and min_radius <= radius <= max_radius and radius > best_radius:
                        best_contour = contour
                        best_radius = radius
                        best_contour_index = idx
                        best_area = cv.contourArea(contour)



                        
                if best_contour is not None and best_radius is not None:
                    (b_x, b_y), b_radius = cv.minEnclosingCircle(best_contour)

                    b_x = int(b_x)
                    b_y = int(b_y)
                    b_radius = int(b_radius)


                    sphero_bolt.finishline_center = (b_x, b_y)
                    sphero_bolt.finishline_radius = b_radius


                    cv.circle(original_frame_copy, sphero_bolt.finishline_center, sphero_bolt.finishline_radius, 
                              COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                    
                    cv.putText(original_frame_copy, sphero_bolt.username, (b_x, b_y-2*sphero_bolt.finishline_radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                        


                    if debug:
                        print(f"Finishline, area: {best_area}, x: {b_x}, y: {b_y}, radius: {b_radius}, total contours: {total_contours}, best contour index: {best_contour_index}")

            

            if start_line is not None:
                cv.line(original_frame_copy, start_line[0], start_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)

                if sphero_bolt.finishline_center is not None and sphero_bolt.finishline_center[1] > start_line[0][1]:
                    if not sphero_bolt.is_started:
                        sphero_bolt.is_started = True
                        sphero_bolt.start_time = datetime.now()
                        print(f"{sphero_bolt.color.value} started, Start Time: {sphero_bolt.start_time.strftime('%H:%M:%S')} sec")



            if finish_line is not None:
                cv.line(original_frame_copy, finish_line[0], finish_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)

                if sphero_bolt.finishline_center is not None and sphero_bolt.finishline_center[1] > finish_line[0][1]:
                    if not sphero_bolt.is_finished and sphero_bolt.is_started:
                        sphero_bolt.is_finished = True
                        sphero_bolt.finish_time = datetime.now()
                        sphero_bolt.lap_time = (sphero_bolt.finish_time - sphero_bolt.start_time).total_seconds()
                        print(f"{sphero_bolt.color.value} finished, Finish Time: {sphero_bolt.finish_time.strftime('%H:%M:%S')} sec")
                        print(f"{sphero_bolt.color.value} Lap Time: {sphero_bolt.lap_time} sec")



            return original_frame_copy
    





