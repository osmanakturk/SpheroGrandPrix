import cv2 as cv
import numpy as np

def trackbar_callback(val):
    pass



def main(test:bool = False):
    cap = cv.VideoCapture(1, cv.CAP_DSHOW)

    HEIGHT = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    WIDTH = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))



    camera = "Camera"
    trackbar = "Trackbar"

    cv.namedWindow(trackbar, cv.WINDOW_NORMAL)
    cv.namedWindow(camera, cv.WINDOW_KEEPRATIO)


    cv.createTrackbar("a_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("a_y", trackbar, 0, HEIGHT, trackbar_callback)

    cv.createTrackbar("bl_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("bl_y", trackbar, 0, HEIGHT, trackbar_callback)

    cv.createTrackbar("cl_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("cl_y", trackbar, 0, HEIGHT, trackbar_callback)

    cv.createTrackbar("b_c_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("br_y", trackbar, 0, HEIGHT, trackbar_callback)
    cv.createTrackbar("cr_y", trackbar, 0, HEIGHT, trackbar_callback)

    



    cv.createTrackbar("dl_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("dl_y", trackbar, 0, HEIGHT, trackbar_callback)

    cv.createTrackbar("dr_x", trackbar, 0, WIDTH, trackbar_callback)
    cv.createTrackbar("dr_y", trackbar, 0, HEIGHT, trackbar_callback)



    if not cap.isOpened():
        exit()

    while True:
        ok, frame = cap.read()

        
        


        if not ok:
            break

        a_x = cv.getTrackbarPos("a_x", trackbar)
        a_y = cv.getTrackbarPos("a_y", trackbar)

        bl_x = cv.getTrackbarPos("bl_x", trackbar)
        bl_y = cv.getTrackbarPos("bl_y", trackbar)

        cl_x = cv.getTrackbarPos("cl_x", trackbar)
        cl_y = cv.getTrackbarPos("cl_y", trackbar)

        b_c_x = cv.getTrackbarPos("b_c_x", trackbar)

        br_x = bl_x + b_c_x
        br_y = cv.getTrackbarPos("br_y", trackbar)

        cr_x = cl_x + b_c_x
        cr_y = cv.getTrackbarPos("cr_y", trackbar)

        cm_x = (cl_x + cr_x)//2
        cm_y = (cl_y + cr_y)//2

        dl_x = cv.getTrackbarPos("dl_x", trackbar)
        dl_y = cv.getTrackbarPos("dl_y", trackbar)

        dr_x = cv.getTrackbarPos("dr_x", trackbar)
        dr_y = cv.getTrackbarPos("dr_y", trackbar)

        


        cv.putText(frame, "A", (a_x, a_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "BL", (bl_x, bl_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "BR", (br_x, br_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "CL", (cl_x, cl_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "CR", (cr_x, cr_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "DL", (dl_x, dl_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)
        cv.putText(frame, "DR", (dr_x, dr_y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv.LINE_AA)



        cv.line(frame, (a_x, a_y), (bl_x, bl_y), (0, 0, 255), 1, cv.LINE_AA)
        

        cv.line(frame, (a_x, a_y), (br_x, br_y), (0, 0, 255), 1, cv.LINE_AA)

        cv.line(frame, (bl_x, bl_y), (cl_x, cl_y), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(frame, (br_x, br_y), (cr_x, cr_y), (0, 0, 255), 1, cv.LINE_AA)

        cv.line(frame, (cl_x, cl_y), (cr_x, cr_y), (0, 0, 255), 1, cv.LINE_AA)

        cv.line(frame, (a_x, a_y), (dl_x, dl_y), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(frame, (a_x, a_y), (dr_x, dr_y), (0, 0, 255), 1, cv.LINE_AA)
        cv.line(frame, (dl_x, dl_y), (dr_x, dr_y), (0, 0, 255), 1, cv.LINE_AA)
        

        if all([a_x, a_y, bl_x, bl_y, br_x, br_y, cl_x, cl_y, cr_x, cr_y, dl_x, dl_y, cm_x, cm_y]):
            overlay = frame.copy()

            back_points = np.array([
                [a_x, a_y], 
                [cm_x, cm_y], 
                [dr_x, dr_y]
            ])

            middle_points = np.array([
                [a_x, a_y], 
                [bl_x, bl_y], 
                [cl_x, cl_y], 
                [cr_x, cr_y], 
                [br_x, br_y] 
                ])
            
            front_points = np.array([
                [a_x, a_y], 
                [dl_x, dl_y], 
                [cm_x, cm_y]     
            ])


            cv.fillPoly(overlay, [back_points], (0, 0, 255))
            cv.fillPoly(overlay, [middle_points], (0, 255, 0))
            cv.fillPoly(overlay, [front_points], (255, 0, 0))
            cv.polylines(overlay, [back_points, middle_points, front_points], True, (0,0,0), 1, cv.LINE_AA)
            cv.imshow("overlay", overlay)
            frame = cv.addWeighted(frame, 0.5, overlay, 0.5, 0)
            



        if test:

            test_overlay = frame.copy()

            test_points1 = np.array(((296, 65), (353, 381), (395, 90)))
            
            test_points2 = np.array(((296, 65), (226, 120), (246, 366), (460, 396), (440, 140)))

            test_points3 = np.array(((296, 65), (310, 480), (353, 381)))
            
            cv.fillPoly(test_overlay, [test_points1], (255, 0, 0))
            cv.fillPoly(test_overlay, [test_points2], (0, 0, 255))
            cv.fillPoly(test_overlay, [test_points3], (255, 0, 0))
            cv.polylines(test_overlay, [test_points1, test_points2, test_points3], True, (0,0,0), 1, cv.LINE_AA)
            cv.imshow("Test Overlay", test_overlay)
            frame = cv.addWeighted(frame, 0.5, test_overlay, 0.5, 0)
            


        cv.imshow(camera, frame)


        key = cv.waitKey(1)

        if key & 0xFF == 27:
            break

        elif key == ord("c"):
            print("*"*20)
            print(f"Back Points : ({a_x, a_y}, {cm_x, cm_y}, {dr_x, dr_y})")
            print(f"Middle Points : ({a_x, a_y}, {bl_x, bl_y}, {cl_x, cl_y}, {cr_x, cr_y}, {br_x, br_y})")
            print(f"Front Points : ({a_x, a_y}, {dl_x, dl_y}, {cm_x, cm_y})")
            print("*"*20)

    cap.release()
    cv.destroyAllWindows()






if __name__=="__main__":
    main(test=False)