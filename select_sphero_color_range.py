import cv2
import numpy as np

# Globals for mouse callback
center_pt = None      # (x, y) center of circle
current_pt = None     # (x, y) current mouse position
drawing = False       # True while dragging
preview = None        # image to show preview of circle

def mouse_callback(event, x, y, flags, param):
    global center_pt, current_pt, drawing, preview

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start circle: record center point
        center_pt = (x, y)
        drawing = True
        preview = frame.copy()

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # While dragging, update preview with live circle
        preview = frame.copy()
        radius = int(np.hypot(x - center_pt[0], y - center_pt[1]))
        cv2.circle(preview, center_pt, radius, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        # Finish circle: record edge point, stop drawing
        current_pt = (x, y)
        drawing = False
        preview = frame.copy()
        radius = int(np.hypot(x - center_pt[0], y - center_pt[1]))
        cv2.circle(preview, center_pt, radius, (0, 255, 0), 2)

def compute_hsv_range(masked_hsv):
    """
    Given an array of HSV pixels (N×3), compute the
    5th and 95th percentiles per channel to skip outliers.
    """
    pixels = masked_hsv.reshape(-1, 3)
    low  = np.percentile(pixels, 5, axis=0)
    high = np.percentile(pixels, 95, axis=0)
    return low.astype(int), high.astype(int)

# --- Main script ---
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("Failed to capture an image from camera")

preview = frame.copy()
cv2.namedWindow("Select Sphere Circle")
cv2.setMouseCallback("Select Sphere Circle", mouse_callback)

print("1) Click at the sphere center, drag to its edge, release to finalize circle.")
print("2) Press 'c' to compute HSV range, 'r' to reset selection, ESC to exit.")

while True:
    # Choose which image to display
    display = preview if (drawing or current_pt) else frame
    cv2.imshow("Select Sphere Circle", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        # Reset selection
        center_pt = None
        current_pt = None
        drawing = False
        preview = frame.copy()
        print("Selection reset.")

    elif key == ord('c'):
        # Compute HSV range if a circle is defined
        if center_pt and current_pt:
            # Compute radius and build circular mask
            radius = int(np.hypot(current_pt[0] - center_pt[0],
                                  current_pt[1] - center_pt[1]))
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center_pt, radius, 255, -1)

            # Extract only the pixels inside the circle
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            masked_hsv = hsv[mask == 255]

            # Compute percentiles to skip outliers
            low, high = np.percentile(masked_hsv, [5, 95], axis=0).astype(int)

            # Print results
            print("\nComputed HSV range (5th–95th percentile):")
            print(f"lower = [{low[0]}, {low[1]}, {low[2]}]")
            print(f"upper = [{high[0]}, {high[1]}, {high[2]}]")
        else:
            print("No circle defined yet. Draw one first.")
        break

    elif key == 27:
        # ESC: exit without computing
        print("Operation cancelled.")
        break

cv2.destroyAllWindows()