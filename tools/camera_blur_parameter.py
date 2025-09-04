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

            red = cv.bitwise_and(frame, frame, mask=red_mask)
            yellow = cv.bitwise_and(frame, frame, mask=yellow_mask)
            blue = cv.bitwise_and(frame, frame, mask=blue_mask)
            green = cv.bitwise_and(frame, frame, mask=green_mask)

            cv.imshow("Red", red)
            cv.imshow("Yellow", yellow)
            cv.imshow("Blue", blue)
            cv.imshow("Green", green)

            red_median = cv.medianBlur(red, median_k)
            yellow_median = cv.medianBlur(yellow, median_k)
            blue_median = cv.medianBlur(blue, median_k)
            green_median = cv.medianBlur(green, median_k)

            cv.imshow("Red Median", red_median)
            cv.imshow("Yellow Median", yellow_median)
            cv.imshow("Blue Median", blue_median)
            cv.imshow("Green Median", green_median)

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
