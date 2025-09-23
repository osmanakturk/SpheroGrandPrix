# SpheroGrandPrix – Tools Folder Overview

This directory contains utility scripts used in the calibration and visual setup of the Sphero BOLT Grand Prix project. These scripts help developers define camera behavior, color filtering, geometric detection zones, and image pre-processing parameters through interactive OpenCV-based UIs.

---

## Tool Descriptions and Use Cases

### 📷 `camera_configuration.py`
- **Purpose**: Lists and visualizes all available OpenCV camera backends and property settings.
- **Functionality**:
  - Uses `pygrabber` to enumerate connected webcams.
  - Displays property values like resolution, FPS, exposure.
  - Checks writable properties.
- **Use Case**: Initial camera configuration; useful for verifying device capabilities and settings like exposure, focus, etc.

---

### `color_trackbar_red.py`, `color_trackbar_green.py`, `color_trackbar_blue.py`, `color_trackbar_yellow.py`
- **Purpose**: Interactive HSV threshold tuning for each Sphero color.
- **Functionality**:
  - Uses trackbars to adjust hue, saturation, and value ranges.
  - Displays raw input, binary mask, and filtered frame.
  - Pressing `c` prints the current HSV range in copyable format.
- **Use Case**: Fine-tuning color segmentation under real-world lighting.

---

### `frame_manipulation_parameters.py`
- **Purpose**: Adjust image enhancement parameters (CLAHE, blur, morphology) to improve mask quality.
- **Functionality**:
  - Bilateral filter, median blur, CLAHE contrast adjustment.
  - Morphological opening/closing for noise removal.
  - Masks and contours displayed per color.
  - Pressing `c` prints final parameters.
- **Use Case**: Segment noise removal and clean color masking.

---

### `finishline_camera_perspective.py`
- **Purpose**: Perspective calibration and start/stop line definition for finishline camera.
- **Functionality**:
  - Drag points using trackbars to define ROI corners.
  - Start/Stop lines added post-warp for accurate timing.
  - Pressing `c` prints perspective matrix and timing line coordinates.
- **Use Case**: Enables accurate lap timing via camera-based finishline crossing detection.

---

### `select_sphero_color_range.py`
- **Purpose**: Select a region over a sphere using mouse and calculate HSV percentile range.
- **Functionality**:
  - Draw a circle using click-drag.
  - Computes 5th to 95th percentile range of HSV values.
- **Use Case**: Quick and statistically robust HSV range estimation.

---

### `select_sphero_size.py`
- **Purpose**: Measure Sphero ball size and location in camera view.
- **Functionality**:
  - Draw a circle around the ball.
  - Outputs center coordinates, radius, and diameter in pixels.
- **Use Case**: Used for tracking, masking, or validation in segmentation scripts.

---

### `set_status_camera_zone.py`
- **Purpose**: Define zone geometry in the status camera’s view (rear/mid/front).
- **Functionality**:
  - Uses 12 trackbars to define polygon points.
  - Overlays colored zones (red, green, blue) with labels and lines.
  - Pressing `c` prints the points for each zone.
- **Use Case**: Important for behavior detection systems (e.g., warning when Sphero enters restricted zones).

---

## How to Use

Run each script independently via:

```bash
python tools/color_trackbar_red.py
python tools/frame_manipulation_parameters.py
python tools/select_sphero_size.py
```

> Press `c` to **print parameters**, `r` to **reset selections**, and `ESC` to **exit**.

---

## Output Parameters

| Tool | Output |
|------|--------|
| `camera_configuration.py` | Writable properties per camera |
| `color_trackbar_*.py` | HSV ranges (min/max for H, S, V) |
| `frame_manipulation_parameters.py` | Blur, CLAHE, morphology settings |
| `select_sphero_color_range.py` | HSV range using percentile mask |
| `select_sphero_size.py` | Pixel radius and center of ball |
| `finishline_camera_perspective.py` | Perspective points and timing lines |
| `set_status_camera_zone.py` | Zone polygons and their coordinates |

---

## Best Practices

- Warm up cameras before calibrating (discard 10+ frames).
- Keep lighting consistent when calibrating.
- Run `frame_manipulation_parameters.py` before `color_trackbar_*.py` to clean up masks.
- Store extracted values in config files for reproducibility.

---

## File Structure

```
tools/
├── __init__.py
├── camera_configuration.py
├── color_trackbar_red.py
├── color_trackbar_green.py
├── color_trackbar_blue.py
├── color_trackbar_yellow.py
├── finishline_camera_perspective.py
├── frame_manipulation_parameters.py
├── select_sphero_color_range.py
├── select_sphero_size.py
├── set_status_camera_zone.py
└── README.md  ← (this file)
```

---
