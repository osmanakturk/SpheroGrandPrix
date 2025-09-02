import cv2 as cv
import numpy as np
import time


# Global variables
center_pt = None
current_pt = None
drawing = False
preview = None

def mouse_callback(event, x, y, flags, param):
    global center_pt, current_pt, drawing, preview

    if event == cv.EVENT_LBUTTONDOWN:
        # Start drawing: record center point
        center_pt = (x, y)
        drawing = True

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        # Update current mouse position for preview
        preview = frame.copy()
        cv.circle(preview, center_pt, int(np.hypot(x - center_pt[0], y - center_pt[1])), (0, 255, 0), 2)

    elif event == cv.EVENT_LBUTTONUP and drawing:
        # Finish drawing: record final point and radius
        current_pt = (x, y)
        drawing = False
        preview = frame.copy()
        radius = int(np.hypot(current_pt[0] - center_pt[0], current_pt[1] - center_pt[1]))
        cv.circle(preview, center_pt, radius, (0, 255, 0), 2)

# Open camera and grab one frame

cap = cv.VideoCapture(2, cv.CAP_DSHOW)


for _ in range(5):
    ret, frame = cap.read()


if not ret:
    raise RuntimeError("Could not read from camera")

preview = frame.copy()
cv.namedWindow("Select Sphere Size")
cv.setMouseCallback("Select Sphere Size", mouse_callback)

print("Instructions:")
print("  1) Click at the sphere center, drag to its edge, release to set radius.")
print("  2) Press 'c' to confirm and print values, 'r' to reset, ESC to exit without saving.")

while True:
    display = preview if drawing or current_pt else frame
    cv.imshow("Select Sphere Size", display)
    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):
        # Reset selection
        current_pt = None
        center_pt = None
        preview = frame.copy()

    elif key == ord('c'):
        # Confirm selection
        if center_pt and current_pt:
            dx = current_pt[0] - center_pt[0]
            dy = current_pt[1] - center_pt[1]
            radius = int(np.hypot(dx, dy))
            diameter = radius * 2
            print("*"*20)
            print("Sphere parameters:")
            print(f"center_x = {center_pt[0]}")
            print(f"center_y = {center_pt[1]}")
            print(f"radius   = {radius}  # in pixels")
            print(f"diameter = {diameter}  # in pixels")
            print("*"*20)
            center_pt = None
            current_pt = None
            preview = frame.copy()
        else:
            print("No circle defined yet.")
       

    elif key == 27:
        # ESC: exit without printing
        print("Selection cancelled.")
        break

cv.destroyAllWindows()