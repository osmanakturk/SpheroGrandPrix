import cv2 as cv
import numpy as np
from backend.constants import HSV_RANGES_STRICT, HSV_RANGES_WIDE, HSV_RANGES_NORMAL



COLORS = HSV_RANGES_NORMAL

def trackbar(_):
    pass


def main():


    cap = cv.VideoCapture(1, cv.CAP_DSHOW)

    cv.namedWindow("Trackbar", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Camera", cv.WINDOW_KEEPRATIO)

    cv.createTrackbar("bilateral_d", "Trackbar", 1, 50, trackbar)
    cv.createTrackbar("median_k", "Trackbar", 1, 50, trackbar)
    cv.createTrackbar("s_color", "Trackbar", 1, 255, trackbar)
    cv.createTrackbar("s_space", "Trackbar", 1, 255, trackbar)

    cv.createTrackbar("clahe_clip", "Trackbar", 1, 50, trackbar)
    cv.createTrackbar("clahe_grid", "Trackbar", 2, 50, trackbar)

    cv.createTrackbar("morph_ksize", "Trackbar", 5, 50, trackbar)
    cv.createTrackbar("morph_iter", "Trackbar", 1, 10, trackbar)


    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            
        
            bilateral_d = int(cv.getTrackbarPos("bilateral_d", "Trackbar"))
            bilateral_d = bilateral_d if bilateral_d % 2 == 1 else bilateral_d +1
            median_k = int(cv.getTrackbarPos("median_k", "Trackbar"))
            median_k = median_k if median_k % 2 == 1 else median_k + 1
            sigma_color = int(cv.getTrackbarPos("s_color", "Trackbar"))
            sigma_color = sigma_color if sigma_color >= 1 else 1
            sigma_space = int(cv.getTrackbarPos("s_space", "Trackbar"))
            sigma_space = sigma_space if sigma_space >= 1 else 1
            clahe_clip = max(1, cv.getTrackbarPos("clahe_clip", "Trackbar"))
            clahe_grid = max(2, cv.getTrackbarPos("clahe_grid", "Trackbar"))
            morph_kernel_size = cv.getTrackbarPos("morph_ksize", "Trackbar")
            morph_kernel_size = max(1, morph_kernel_size if morph_kernel_size%2==1 else morph_kernel_size+1)
            morph_iterator = max(1, cv.getTrackbarPos("morph_iter", "Trackbar"))





            bilateral_blur = cv.bilateralFilter(frame, bilateral_d, sigma_color, sigma_space)
            



            cv.imshow("Camera", frame)
            cv.imshow("Bilateral Blur", bilateral_blur)


            hsv = cv.cvtColor(bilateral_blur, cv.COLOR_BGR2HSV)

            h, s, v = cv.split(hsv)

            clahe = cv.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(clahe_grid, clahe_grid))
            v_clahe = clahe.apply(v)

            hsv = cv.merge([h, s, v_clahe])

            frame_clahe = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            cv.imshow("Frame Clahe", frame_clahe)

            

            red1_mask = cv.inRange(hsv, COLORS["Red1"]["Lower"], COLORS["Red1"]["Upper"])
            red2_mask = cv.inRange(hsv, COLORS["Red2"]["Lower"], COLORS["Red2"]["Upper"])
            red_mask = cv.bitwise_or(red1_mask, red2_mask)
            

            yellow_mask = cv.inRange(hsv, COLORS["Yellow"]["Lower"], COLORS["Yellow"]["Upper"])


            blue_mask = cv.inRange(hsv, COLORS["Blue"]["Lower"], COLORS["Blue"]["Upper"])



            green_mask = cv.inRange(hsv, COLORS["Green"]["Lower"], COLORS["Green"]["Upper"])




            cv.imshow("Red Mask", red_mask)
            cv.imshow("Yellow Mask", yellow_mask)
            cv.imshow("Blue Mask", blue_mask)
            cv.imshow("Green Mask", green_mask)

            morph_ellipse_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))


            red_mask_median = cv.medianBlur(red_mask, median_k)

            red_mask_median_morph = cv.morphologyEx(red_mask_median, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            red_mask_median_morph = cv.morphologyEx(red_mask_median_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)

            yellow_mask_median = cv.medianBlur(yellow_mask, median_k)
            yellow_mask_median_morph = cv.morphologyEx(yellow_mask_median, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            yellow_mask_median_morph = cv.morphologyEx(yellow_mask_median_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)

            blue_mask_median = cv.medianBlur(blue_mask, median_k)
            blue_mask_median_morph = cv.morphologyEx(blue_mask_median, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            blue_mask_median_morph = cv.morphologyEx(blue_mask_median_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)


            green_mask_median = cv.medianBlur(green_mask, median_k)
            green_mask_median_morph = cv.morphologyEx(green_mask_median, cv.MORPH_OPEN, morph_ellipse_kernel, iterations=morph_iterator)
            green_mask_median_morph = cv.morphologyEx(green_mask_median_morph, cv.MORPH_CLOSE, morph_ellipse_kernel, iterations=morph_iterator)


            cv.imshow("Red Mask Median", red_mask_median)
            cv.imshow("Yellow Mask Median", yellow_mask_median)
            cv.imshow("Blue Mask Median", blue_mask_median)
            cv.imshow("Green Mask Median", green_mask_median)


            cv.imshow("Red Mask Median Morphology", red_mask_median_morph)
            cv.imshow("Yellow Mask Median Morphology", yellow_mask_median_morph)
            cv.imshow("Blue Mask Median Morphology", blue_mask_median_morph)
            cv.imshow("Green Mask Median Morphology", green_mask_median_morph)



            red = cv.bitwise_and(frame, frame, mask=red_mask_median_morph)
            yellow = cv.bitwise_and(frame, frame, mask=yellow_mask_median_morph)
            blue = cv.bitwise_and(frame, frame, mask=blue_mask_median_morph)
            green = cv.bitwise_and(frame, frame, mask=green_mask_median_morph)

            cv.imshow("Red", red)
            cv.imshow("Yellow", yellow)
            cv.imshow("Blue", blue)
            cv.imshow("Green", green)


            green_contours_frame = frame.copy()
            yellow_contours_frame = frame.copy()
            red_contours_frame = frame.copy()
            blue_contours_frame = frame.copy()


            green_contours, _ = cv.findContours(green_mask_median_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(green_contours_frame, green_contours, -1, (0, 0, 255), 3)

            yellow_contours, _ = cv.findContours(yellow_mask_median_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(yellow_contours_frame, yellow_contours, -1, (0, 0, 255), 3)
            
            red_contours, _ = cv.findContours(red_mask_median_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(red_contours_frame, red_contours, -1, (0, 0, 255), 3)
            
            blue_contours, _ = cv.findContours(blue_mask_median_morph, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(blue_contours_frame, blue_contours, -1, (0, 0, 255), 3)
    
            
            cv.imshow("Green Contours", green_contours_frame)
            cv.imshow("Yellow Contours", yellow_contours_frame)
            cv.imshow("Red Contours", red_contours_frame)
            cv.imshow("Blue Contours", blue_contours_frame)



            key = cv.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord("c"):
                print("*"*20)
                print(f"Bilateral Diameter: {bilateral_d}")
                print(f"Bilateral Sigma Color: {sigma_color}")
                print(f"Bilateral Sigma Space: {sigma_space}")
                print(f"Median Kernel Size: {median_k}")
                print(f"Clahe Clip Limit Size: {clahe_clip}.0")
                print(f"Clahe Tile Grid Size: {clahe_grid}")
                print(f"Morphology Kernel Size: {morph_kernel_size}")
                print(f"Morphology Iterator: {morph_iterator}")
                print("*"*20)
    finally:
        cap.release()
        cv.destroyAllWindows()
    

if __name__=="__main__":
    main()
