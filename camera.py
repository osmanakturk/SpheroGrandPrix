import numpy as np
import cv2 as cv




#TODO: FIND CAMERA INDEX
cap = cv.VideoCapture(1, cv.CAP_DSHOW) 

FRAME_HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
FRAME_WIDTH = cap.get(cv.CAP_PROP_FRAME_WIDTH)

print(f"FRAME_HEIGHT: {FRAME_HEIGHT}, FRAME_HEIGHT: {FRAME_WIDTH}")



while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

  
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break


print("Camera closing")
cap.release()
cv.destroyAllWindows()