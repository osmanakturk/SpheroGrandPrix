import cv2

# Global variables
ref_point = []
cropping = False
preview_image = None

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image, preview_image

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE and cropping:
        # Show dynamic rectangle while dragging
        preview_image = image.copy()
        cv2.rectangle(preview_image, ref_point[0], (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        preview_image = image.copy()

# Open camera and get single frame
cap = cv2.VideoCapture(0)
ret, image = cap.read()
cap.release()

preview_image = image.copy()
cv2.namedWindow("Select Finish Line")
cv2.setMouseCallback("Select Finish Line", click_and_crop)

while True:
    # Show either the static or live preview
    if cropping:
        cv2.imshow("Select Finish Line", preview_image)
    else:
        cv2.imshow("Select Finish Line", image)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = preview_image = image.copy()
        ref_point = []

    elif key == ord("c"):
        break

    elif key == 27:  # ESC
        ref_point = []
        break

cv2.destroyAllWindows()

# Print results
if len(ref_point) == 2:
    (x1, y1) = ref_point[0]
    (x2, y2) = ref_point[1]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    print("\nCopy these values into your main script:")
    print(f"x1 = {x1}")
    print(f"y1 = {y1}")
    print(f"x2 = {x2}")
    print(f"y2 = {y2}")
    print(f"roi = frame[{y1}:{y2}, {x1}:{x2}]")
else:
    print("No region selected.")