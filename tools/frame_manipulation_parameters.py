import cv2 as cv
import numpy as np
from backend.constants import COLOR_RANGES_STRICT, COLOR_RANGES_WIDE, COLOR_RANGES_NORMAL



COLORS = COLOR_RANGES_WIDE

def trackbar(_):
    pass


def main():


    cap = cv.VideoCapture(1, cv.CAP_DSHOW)

    cv.namedWindow("Trackbar", cv.WINDOW_NORMAL)
    cv.namedWindow("Camera", cv.WINDOW_KEEPRATIO)

    cv.createTrackbar("briteral_d", "Trackbar", 1, 255, trackbar)
    cv.createTrackbar("median_k", "Trackbar", 1, 255, trackbar)
    cv.createTrackbar("s_color", "Trackbar", 1, 255, trackbar)
    cv.createTrackbar("s_space", "Trackbar", 1, 255, trackbar)



    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            
        
            briteral_d = int(cv.getTrackbarPos("briteral_d", "Trackbar"))
            briteral_d = briteral_d if briteral_d % 2 == 1 else briteral_d +1
            median_k = int(cv.getTrackbarPos("median_k", "Trackbar"))
            median_k = median_k if median_k % 2 == 1 else median_k + 1
            sigma_color = int(cv.getTrackbarPos("s_color", "Trackbar"))
            sigma_color = sigma_color if sigma_color >= 1 else 1
            sigma_space = int(cv.getTrackbarPos("s_space", "Trackbar"))
            sigma_space = sigma_space if sigma_space >= 1 else 1



            bilateral_blur = cv.bilateralFilter(frame, briteral_d, sigma_color, sigma_space)
            



            cv.imshow("Camera", frame)
            cv.imshow("Bilateral Blur", bilateral_blur)


            hsv = cv.cvtColor(bilateral_blur, cv.COLOR_BGR2HSV)
            hsv[:, :, 2] = cv.equalizeHist(hsv[:, :, 2])


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


            red_mask_median = cv.medianBlur(red_mask, median_k)
            yellow_mask_median = cv.medianBlur(yellow_mask, median_k)
            blue_mask_median = cv.medianBlur(blue_mask, median_k)
            green_mask_median = cv.medianBlur(green_mask, median_k)

            cv.imshow("Red Mask Median", red_mask_median)
            cv.imshow("Yellow Mask Median", yellow_mask_median)
            cv.imshow("Blue Mask Median", blue_mask_median)
            cv.imshow("Green Mask Median", green_mask_median)



            red = cv.bitwise_and(frame, frame, mask=red_mask_median)
            yellow = cv.bitwise_and(frame, frame, mask=yellow_mask_median)
            blue = cv.bitwise_and(frame, frame, mask=blue_mask_median)
            green = cv.bitwise_and(frame, frame, mask=green_mask_median)

            cv.imshow("Red", red)
            cv.imshow("Yellow", yellow)
            cv.imshow("Blue", blue)
            cv.imshow("Green", green)


            green_contours_frame = frame.copy()
            yellow_contours_frame = frame.copy()
            red_contours_frame = frame.copy()
            blue_contours_frame = frame.copy()


            green_contours, green_hierarchy = cv.findContours(green_mask_median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(green_contours_frame, green_contours, -1, (0, 0, 255), 3)

            yellow_contours, yellow_hierarchy = cv.findContours(yellow_mask_median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(yellow_contours_frame, yellow_contours, -1, (0, 0, 255), 3)
            
            red_contours, red_hierarchy = cv.findContours(red_mask_median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(red_contours_frame, red_contours, -1, (0, 0, 255), 3)
            
            blue_contours, blue_hierarchy = cv.findContours(blue_mask_median, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
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
                print(f"Briteral Diameter: {briteral_d}")
                print(f"Briteral Sigma Color: {sigma_color}")
                print(f"Briteral Sigma Space: {sigma_space}")
                print(f"Median Kernel Size: {median_k}")
                print("*"*20)
    finally:
        cap.release()
        cv.destroyAllWindows()
    

if __name__=="__main__":
    main()
