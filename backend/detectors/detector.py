import numpy as np
import cv2 as cv
from typing import Optional, Tuple
from datetime import datetime
from backend.constants import COLORS_BGR
from backend.enums import SpheroColor
from backend.configs import DetectorConfig





class Detector:
        
        
    @staticmethod
    def get_detected_path_frame(config: DetectorConfig) -> Optional[cv.typing.MatLike]:
            
            sphero_bolt = config.sphero_bolt
            HSV_RANGES = config.hsv_ranges.value

            min_radius = config.min_radius
            max_radius = config.max_radius


            if sphero_bolt.path_frame is None:
                return None

            bilateral_diameter = max(1, config.bilateral_diameter if config.bilateral_diameter%2 == 1 else config.bilateral_diameter+1)
            bilateral_sigma_color = max(1, config.bilateral_sigma_color)
            bilateral_sigma_space = max(1, config.bilateral_sigma_space)
            median_kernel_size = max(1, config.median_kernel_size if config.median_kernel_size%2 == 1 else config.median_kernel_size + 1)
            morph_kernel_size = max(1, config.morph_kernel_size if config.morph_kernel_size%2==1 else config.morph_kernel_size+1)
            morph_iterator = max(1, config.morph_iterator)
            clahe_clip_limit = max(1.0, config.clahe_clip_limit)
            clahe_tile_grid_size = max(2, config.clahe_tile_grid_size)


            
            
            path_frame = sphero_bolt.path_frame.copy()

            response = np.zeros_like(path_frame, np.uint8)

            path_bilateral = cv.bilateralFilter(path_frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if config.debug:
                cv.imshow(f"Path Bilateral {sphero_bolt.color.value}", path_bilateral)

       

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

            
            if config.contours_chain_approx_simple:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            if config.debug:
                masked_frame = cv.bitwise_and(path_frame, path_frame, mask=median_mask_morph)
                cv.imshow(f"{sphero_bolt.color.value} Masked", masked_frame)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask", median_mask)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask Morphology", median_mask_morph)
                contours_frame = cv.drawContours(path_frame.copy(), contours, -1, (0, 0, 255), 3)
                cv.imshow(f"{sphero_bolt.color.value} Contours", contours_frame)
                cv.waitKey(1)
            


            if contours:

                best_contour = None
                best_radius = -1
                best_contour_index = 0
                total_contours = len(contours)
                best_area = 0.0

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

                    
                    cv.circle(response, sphero_bolt.path_center, sphero_bolt.path_radius, 
                              COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                    

            
                    if sphero_bolt.path_canvas is not None:
                        cv.line(sphero_bolt.path_canvas, sphero_bolt.path_previous_center, sphero_bolt.path_center, 
                                COLORS_BGR[sphero_bolt.color.value], 3, cv.LINE_AA)
                    
                    sphero_bolt.path_previous_center = (b_x, b_y)

                    cv.putText(response, sphero_bolt.username, (b_x, b_y-2*sphero_bolt.path_radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                        
                    if config.debug:
                        print(f"Path {sphero_bolt.color.value}, area: {best_area}, x: {b_x}, y: {b_y}, radius: {b_radius}, total contours: {total_contours}, best contour index: {best_contour_index}")

            if sphero_bolt.path_canvas is not None:
                response = cv.bitwise_or(response, sphero_bolt.path_canvas)


            return response
        




    @staticmethod
    def get_detected_finishline_frame(
        config:DetectorConfig, 
        start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
        finish_line:Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
        ) -> Optional[cv.typing.MatLike]:
            
            sphero_bolt = config.sphero_bolt
            HSV_RANGES = config.hsv_ranges.value
            min_radius = config.min_radius
            max_radius = config.max_radius



            if sphero_bolt.finishline_frame is None:
                return None
            


            bilateral_diameter = max(1, config.bilateral_diameter if config.bilateral_diameter%2 == 1 else config.bilateral_diameter+1)
            bilateral_sigma_color = max(1, config.bilateral_sigma_color)
            bilateral_sigma_space = max(1, config.bilateral_sigma_space)
            median_kernel_size = max(1, config.median_kernel_size if config.median_kernel_size%2 == 1 else config.median_kernel_size + 1)
            morph_kernel_size = max(1, config.morph_kernel_size if config.morph_kernel_size%2==1 else config.morph_kernel_size+1)
            morph_iterator = max(1, config.morph_iterator)
            clahe_clip_limit = max(1.0, config.clahe_clip_limit)
            clahe_tile_grid_size = max(2, config.clahe_tile_grid_size)

            

            finishline_frame = sphero_bolt.finishline_frame.copy()
        
            response = np.zeros_like(finishline_frame, np.uint8)

            finishline_bilateral = cv.bilateralFilter(finishline_frame, bilateral_diameter, bilateral_sigma_color, bilateral_sigma_space)


            if config.debug:
                cv.imshow(f"Finishline Bilateral {sphero_bolt.color.value}", finishline_bilateral)

       

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


            if config.contours_chain_approx_simple:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            else:
                contours, _ = cv.findContours(median_mask_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)


            if config.debug:
                masked_frame = cv.bitwise_and(finishline_frame, finishline_frame, mask=median_mask_morph)
                cv.imshow(f"{sphero_bolt.color.value} Masked", masked_frame)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask", median_mask)
                cv.imshow(f"{sphero_bolt.color.value} Median Mask Morphology", median_mask_morph)
                contours_frame = cv.drawContours(finishline_frame.copy(), contours, -1, (0, 0, 255), 3)
                cv.imshow(f"{sphero_bolt.color.value} Contours", contours_frame)
                cv.waitKey(1)

            
           

            if contours:

                best_contour = None
                best_radius = -1
                best_contour_index = 0
                total_contours = len(contours)
                best_area = 0.0
                


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

                    if sphero_bolt.finishline_previous_center is None:
                        sphero_bolt.finishline_previous_center = (b_x, b_y)



                    
                    cv.circle(response, sphero_bolt.finishline_center, sphero_bolt.finishline_radius, 
                              COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                    

                    cv.putText(response, sphero_bolt.username, (b_x, b_y-2*sphero_bolt.finishline_radius), 
                            cv.FONT_HERSHEY_COMPLEX_SMALL, 1, COLORS_BGR[sphero_bolt.color.value], 2, cv.LINE_AA)
                    
                    
                    if sphero_bolt.finishline_canvas is not None:
                        if (start_line[0][0] <= sphero_bolt.finishline_center[0] <= start_line[1][0] and start_line[0][0] <= sphero_bolt.finishline_previous_center[0] <= start_line[1][0]) or (finish_line[0][0] <= sphero_bolt.finishline_center[0] <= finish_line[1][0] and finish_line[0][0] <= sphero_bolt.finishline_previous_center[0] <= finish_line[1][0]):
                            cv.line(sphero_bolt.finishline_canvas, sphero_bolt.finishline_previous_center, sphero_bolt.finishline_center, 
                                    COLORS_BGR[sphero_bolt.color.value], 3, cv.LINE_AA)

                    if config.debug:
                        cv.waitKey(1)
                        print(f"Finishline {sphero_bolt.color.value}, area: {best_area}, x: {b_x}, y: {b_y}, radius: {b_radius}, total contours: {total_contours}, best contour index: {best_contour_index}")
            

            if start_line is not None:

                if sphero_bolt.finishline_center is not None: 

                    
                    if start_line[0][1] > sphero_bolt.finishline_center[1] and start_line[1][0] > sphero_bolt.finishline_center[0]:
                    
                            if not sphero_bolt.is_started and sphero_bolt.is_lap_started:
                                sphero_bolt.is_started = True
                                sphero_bolt.start_time = datetime.now()
                                print(f"{sphero_bolt.color.value} started, Start Time: {sphero_bolt.start_time.strftime('%H:%M:%S')} sec")



            if finish_line is not None:

                if sphero_bolt.finishline_center is not None:
                    

                    if finish_line[0][1] > sphero_bolt.finishline_center[1] and sphero_bolt.finishline_center[0] > finish_line[0][0]:
                    
                            if not sphero_bolt.is_finished and sphero_bolt.is_started:
                                sphero_bolt.is_finished = True
                                sphero_bolt.finish_time = datetime.now()
                                sphero_bolt.total_lap_time = (sphero_bolt.finish_time - sphero_bolt.start_time).total_seconds()
                                print(f"{sphero_bolt.color.value} finished, Finish Time: {sphero_bolt.finish_time.strftime('%H:%M:%S')} sec")
                                print(f"{sphero_bolt.color.value} Lap Time: {sphero_bolt.total_lap_time} sec")
                                if sphero_bolt.path_canvas is not None and sphero_bolt.total_lap_time is not None:
                                    sphero_bolt.save_path_img()


            
            sphero_bolt.finishline_previous_center = sphero_bolt.finishline_center

            if sphero_bolt.finishline_canvas is not None:
                response = cv.bitwise_or(response, sphero_bolt.finishline_canvas)

  
            return response
    





