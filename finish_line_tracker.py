import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("./models/yolo11n.pt") 


# Finish line configuration
FINISH_X1 = 800
FINISH_X2 = 1300
FINISH_Y = 598  # horizontal finish line for upward crossings

# HSV color ranges
COLOR_RANGES = {
    'yellow': [(15,  50,  50), (50, 255, 255)],
    'green':  [(35,  40,  40), (90, 255, 255)],
    'blue':   [(85,  40,  40), (140, 255, 255)],
    'red':    [(0,   70,  70), (15, 255, 255)],  
    'purple': [(125, 30,  30), (170, 255, 255)]
}

# State tracking
color_states = {}
color_prev_y = {}
color_start_times = {}
color_lap_times = {}

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(mask, (FINISH_X1, 0), (FINISH_X2, frame.shape[0]), 255, -1)
    
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
    frame[np.where(mask == 0)] = 255  # White background for better visibility
    
    
    # tracker="bytetrack.yaml" , tracker="botsort.yaml"
    results = model.track(masked_frame, tracker="bytetrack.yaml", persist=True, verbose=False, stream=True)

    if results:
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = x1 + (x2 - x1) // 2
                center_y = y1 + (y2 - y1) // 2

                if not (FINISH_X1 <= center_x <= FINISH_X2):
                    continue

                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue

                # Apply CLAHE to enhance contrast before HSV conversion
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                v_clahe = clahe.apply(v)
                hsv_clahe = cv2.merge((h, s, v_clahe))

                mask_circle = np.zeros(roi.shape[:2], dtype=np.uint8)
                radius = int(min(x2 - x1, y2 - y1) / 2)
                cv2.circle(mask_circle, (roi.shape[1]//2, roi.shape[0]//2), radius, 255, -1)

                counts = {}
                for name, (lower, upper) in COLOR_RANGES.items():
                    mask = cv2.inRange(hsv_clahe, np.array(lower), np.array(upper))
                    mask = cv2.bitwise_and(mask, mask, mask=mask_circle)
                    counts[name] = cv2.countNonZero(mask)

                color_name, max_count = max(counts.items(), key=lambda x: x[1])
                if max_count < np.pi * (radius**2) * 0.1:
                    continue

                prev_y = color_prev_y.get(color_name, center_y)
                color_prev_y[color_name] = center_y

                if color_states.get(color_name) == "done":
                    continue

                if color_states.get(color_name) is None and prev_y > FINISH_Y >= center_y:
                    color_states[color_name] = "started"
                    color_start_times[color_name] = time.time()
                    print(f"[{color_name.upper()}] → Lap started")

                elif color_states.get(color_name) == "started" and prev_y > FINISH_Y >= center_y:
                    lap_time = time.time() - color_start_times[color_name]
                    color_states[color_name] = "done"
                    color_lap_times[color_name] = lap_time
                    print(f"[{color_name.upper()}] → Lap completed in {lap_time:.2f} sec")

                cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.putText(frame, color_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw finish lines
    h = frame.shape[0]
    cv2.line(frame, (FINISH_X1, 0), (FINISH_X1, h), (0, 0, 255), 2)
    cv2.line(frame, (FINISH_X2, 0), (FINISH_X2, h), (0, 0, 255), 2)
    cv2.line(frame, (FINISH_X1, FINISH_Y), (FINISH_X2, FINISH_Y), (0, 0, 255), 4)
    cv2.putText(frame, "FINISH LINE", (FINISH_X1 + 10, FINISH_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Finish Line Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


