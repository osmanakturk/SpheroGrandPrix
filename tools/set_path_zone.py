import cv2 as cv
import numpy as np



drawing = False
x0 = y0 = None
rect = None  # (x1,y1,x2,y2)



def norm_rect(x1, y1, x2, y2):
    xA, xB = sorted([x1, x2])
    yA, yB = sorted([y1, y2])
    return (xA, yA, xB, yB)



def on_mouse(event, x, y, flags, param):
    global drawing, x0, y0, rect
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y
        rect = (x, y, x, y)  
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        rect = norm_rect(x0, y0, x, y)  
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        rect = norm_rect(x0, y0, x, y)  



def draw_panel(img):

    overlay = img.copy()
    cv.rectangle(overlay, (0,0), (240, 100), (0,0,0), -1)
    img[:] = cv.addWeighted(overlay, 0.5, img, 0.5, 0)

    cv.putText(img, "drag: draw rectangle", (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "r: reset   c: confirm", (8, 40), cv.FONT_HERSHEY_SIMPLEX, 0.6, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "ESC: cancel", (8, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (24,221,245), 1, cv.LINE_AA)

    


#cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture("./tests/sphero1_270.mp4")

ret, frame = cap.read()
    

if not ret:
    exit()

main_window = "Draw One Rectangle"

cv.namedWindow(main_window, cv.WINDOW_NORMAL)
cv.setMouseCallback(main_window, on_mouse)

while True:
    display = frame.copy()
    draw_panel(display)


    if rect is not None:
        x1,y1,x2,y2 = rect
        cv.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 2, cv.LINE_AA)

    cv.imshow(main_window, display)
    k = cv.waitKey(10) & 0xFF
    if k == ord('r'):
        rect = None
    elif k == ord('c'):
        break
    elif k == 27:  # ESC
        rect = None
        break

cap.release()
cv.destroyAllWindows()

if rect is not None:
    x1,y1,x2,y2 = rect
    print("\nCopy these values into your script:")
    print(f"RECT = (x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2})")
else:
    print("No rectangle selected.")

