import cv2 as cv

def main():
    #cap = cv.VideoCapture("v4l2src device=/dev/video0 ! videoconvert ! appsink", cv.CAP_GSTREAMER)
    # sudo apt install -y gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav v4l-utils
    
    cam_api = cv.CAP_DSHOW
    cam1_index = 1
    cam2_index = 2

    try:
        cap1 = cv.VideoCapture(cam1_index, cam_api)
    except Exception as e:
        print(f"Camera(1) VideoCapture Error: {e}")
        return

    try:
        cap2 = cv.VideoCapture(cam2_index, cam_api)
    except Exception as e:
        print(f"Camera(2) VideoCapture Error: {e}")
        return

    if not cap1.isOpened():
        print("Camera(1) could not be opened")
        return
    if not cap2.isOpened():
        print("Camera(2) could not be opened")
        return

    cv.namedWindow("webcam1", cv.WINDOW_KEEPRATIO)
    cv.namedWindow("webcam2", cv.WINDOW_KEEPRATIO)

    while True:
        ok1, ok2 = False, False
        frame1, frame2 = None, None

        try:
            ok1, frame1 = cap1.read()
        except Exception as e:
            print(f"Camera(1) Read Error: {e}")

        try:
            ok2, frame2 = cap2.read()
        except Exception as e:
            print(f"Camera(2) Read Error: {e}")

        if ok1:
            cv.imshow("webcam1", frame1)
        else:
            print("Camera(1) could not be read")

        if ok2:
            cv.imshow("webcam2", frame2)
        else:
            print("Camera(2) could not be read")

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
