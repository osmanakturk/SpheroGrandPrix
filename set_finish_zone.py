import cv2 as cv
import numpy as np



FINISH = {"X1": None, "X2": None, "Y": None}
MODE   = {"X1": False, "X2": False, "Y": False}
mx, my = None, None 




def click_and_crop(event, x, y, flags, param):
    global mx, my, FINISH, MODE
    mx, my = x, y

    if event == cv.EVENT_LBUTTONDOWN:
        if MODE["X1"]:
            FINISH["X1"] = x
            print("X1 set ->", x)
        elif MODE["X2"]:
            FINISH["X2"] = x
            print("X2 set ->", x)
        elif MODE["Y"]:
            FINISH["Y"]  = y
            print("Y set  ->", y)




def draw_ui(img):
    panel_h, panel_w = 140, 260
    overlay = img.copy()
    cv.rectangle(overlay, (0,0), (panel_w, panel_h), (0,0,0), -1)
    img[:] = cv.addWeighted(overlay, 0.5, img, 0.5, 0)

    cv.putText(img, "press a: set X1 line", (8, 20),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "press z: set X2 line", (8, 40),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "press e: set Y line ", (8, 60),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "press r: reset lines", (8, 80),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "press c: continue   ", (8,100),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)
    cv.putText(img, "press ESC: cancel   ", (8,120),  cv.FONT_HERSHEY_SIMPLEX, 0.5, (24,221,245), 1, cv.LINE_AA)




def draw_lines(img, color=(0,0,255), thick=3):

    h, w = img.shape[:2]
    if FINISH["X1"] is not None:
        x = int(FINISH["X1"])
        cv.line(img, (x, 0), (x, h), color, thick, cv.LINE_AA)
    if FINISH["X2"] is not None:
        x = int(FINISH["X2"])
        cv.line(img, (x, 0), (x, h), color, thick, cv.LINE_AA)
    if FINISH["Y"]  is not None:
        y = int(FINISH["Y"])
        cv.line(img, (0, y), (w, y), color, thick, cv.LINE_AA)



def draw_live_preview(img, color=(0,255,255), thick=2):

    global mx, my
    if mx is None or my is None:
        return
    h, w = img.shape[:2]
    if MODE["X1"] or MODE["X2"]:
        cv.line(img, (mx, 0), (mx, h), color, thick, cv.LINE_AA)
    elif MODE["Y"]:
        cv.line(img, (0, my), (w, my), color, thick, cv.LINE_AA)




#cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap = cv.VideoCapture("./sphero1_270.mp4")


ret, base_image = cap.read()

cap.release()


if not ret:
    exit()

window = "Select Finish Line"
cv.namedWindow(window, cv.WINDOW_KEEPRATIO)
cv.setMouseCallback(window, click_and_crop)

while True:

    display = base_image.copy()

    draw_ui(display)       
    draw_lines(display)      
    draw_live_preview(display)  


    active = [k for k,v in MODE.items() if v]
    active_str = f"ACTIVE: {active[0]}" if active else "ACTIVE: none"

    cv.putText(display, active_str, (display.shape[1]-150, 25), cv.FONT_HERSHEY_SIMPLEX, 0.6, (50,240,50), 2, cv.LINE_AA)

    cv.imshow(window, display)
    key = cv.waitKey(10) & 0xFF

    if key == ord('a'):
        MODE["X1"], MODE["X2"], MODE["Y"] = True, False, False
    elif key == ord('z'):
        MODE["X1"], MODE["X2"], MODE["Y"] = False, True, False
    elif key == ord('e'):
        MODE["X1"], MODE["X2"], MODE["Y"] = False, False, True
    elif key == ord('r'):
        FINISH = {"X1": None, "X2": None, "Y": None}
        MODE   = {"X1": False, "X2": False, "Y": False}
        mx, my = None, None
    elif key == ord('c'):
        break
    elif key == 27:  # ESC
        FINISH = {"X1": None, "X2": None, "Y": None}
        break

cv.destroyAllWindows()



if all(v is not None for v in FINISH.values()):
    print("\nCopy these values into your main script:")
    print(f"FINISH_X1 = {FINISH['X1']}")
    print(f"FINISH_X2 = {FINISH['X2']}")
    print(f"FINISH_Y  = {FINISH['Y']}")
else:
    print("No region selected.")
