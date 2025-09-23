# SpheroGrandPrix — `tools/` Utilities

This folder contains **calibration, camera, and perspective utilities** used to set up and debug the Sphero BOLT Grand Prix system (HSV color segmentation, per‑camera perspective rectification, finish‑line ROI placement, and pre‑processing parameter tuning). These scripts are meant to be run **stand‑alone** during track setup and fed back into the main app as configuration values.

> **TL;DR workflow**
> 1) Set camera backend + inspect properties → `camera_configuration.py`  
> 2) Tune pre‑processing (blur/CLAHE/morphology) → `frame_manipulation_parameters.py`  
> 3) Calibrate HSV ranges per color → `color_trackbar_*.py` or `select_sphero_color_range.py`  
> 4) Measure sphere size in pixels → `select_sphero_size.py`  
> 5) Define perspective crops (Path & Finishline cameras) → `path_camera_perspective.py`, `finishline_camera_perspective.py`  
> 6) Paste the printed values into your project’s configs (e.g., `backend/constants.py`, camera configs).

---

## Prerequisites

- Python 3.10+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- **Windows (recommended for these tools):** use **DirectShow** backend (`cv.CAP_DSHOW`).  
  For macOS use `cv.CAP_AVFOUNDATION`; for Linux use `cv.CAP_V4L2` (or `cv.CAP_GSTREAMER`).  
- Optional (Windows only): `pygrabber` for enumerating DirectShow filters (`pip install pygrabber`).

> Cameras should be connected and selected by index in each script (`cv.VideoCapture(index, backend)`).

---

## How to run

From the repository root:

```bash
# Activate your virtualenv if needed

# 1) Explore camera backends & properties (Windows):
python ./tools/camera_configuration.py

# 2) Tune pre-processing and inspect masks:
python ./tools/frame_manipulation_parameters.py

# 3) Calibrate HSV ranges with live trackbars (per color):
python ./tools/color_trackbar_red.py
python ./tools/color_trackbar_yellow.py
python ./tools/color_trackbar_green.py
python ./tools/color_trackbar_blue.py

# 4) Calibrate HSV from a circular ROI over the sphere (percentile-based):
python ./tools/select_sphero_color_range.py

# 5) Measure sphere size (center, radius, diameter in pixels):
python ./tools/select_sphero_size.py

# 6) Define perspective crops for path & finishline cameras:
python ./tools/path_camera_perspective.py
python ./tools/finishline_camera_perspective.py
```

Most tools print the **final parameters to the console** when you press **`c`**. Press **`ESC`** to exit.

---

## Tool‑by‑Tool Reference

### 1) `camera_configuration.py`
**Purpose.** Inspect available **OpenCV camera backends** and **CAP_PROP settings**, draw selected property values on a preview frame, and quickly test **which properties are writable** on your device/backends.

**Highlights.**
- `ALL_CAPS`: curated list of OpenCV backends and `CAP_PROP_*` keys you can iterate over.
- `find_all_caps()`: dumps available `cv.CAP_*` symbols (useful when upgrading OpenCV).
- `is_cap_writable(cap, values)`: checks which properties can be set on the current `VideoCapture`.
- `draw_info(cap, img, values)`: overlays property values onto an enlarged canvas.

**Typical outputs.**
- Backend selection (e.g., `cv.CAP_DSHOW`, `cv.CAP_AVFOUNDATION`, `cv.CAP_V4L2`).
- Writable camera properties per device (e.g., FPS, exposure, focus, white balance).
- Live preview with a right‑hand panel of current property values.

**When to use.** First step of track setup to pick a **stable backend** and confirm key **camera controls** are writable for your environment.

---

### 2) `frame_manipulation_parameters.py`
**Purpose.** Interactive **pre‑processing tuner** for color segmentation. Lets you adjust **bilateral filter**, **median blur**, **CLAHE (on V channel)**, and **morphology** parameters, while viewing the resulting **HSV masks** and **contours** for each Sphero color.

**Trackbars.**
- `bilateral_d`, `s_color`, `s_space` — bilateral filter diameter & sigmas.  
- `median_k` — median blur kernel (auto‑snapped to odd).  
- `clahe_clip`, `clahe_grid` — contrast equalization on **V channel** (tile grid, clip limit).  
- `morph_ksize`, `morph_iter` — ellipse kernel size & iteration count for open/close.

**Views.**
- Raw camera, bilateral output, CLAHE output.
- Raw and morphed binary masks for **Red** (two ranges), **Yellow**, **Blue**, **Green**.
- Contour overlays per color and masked color extracts.

**Typical outputs.**
- A stable set of pre‑processing parameters to copy into your detector config:
  ```python
  bilateral_d=9, sigma_color=75, sigma_space=75,
  median_k=5, clahe_clip=3, clahe_grid=8,
  morph_kernel=9, morph_iter=2
  ```

**When to use.** Before HSV tuning; get smooth masks under your lighting so HSV ranges become tighter and more robust.

---

### 3) `color_trackbar_red.py`, `color_trackbar_yellow.py`, `color_trackbar_green.py`, `color_trackbar_blue.py`
**Purpose.** Live **HSV range calibration** per color with keyboard‑to‑console export.

**How it works.**
- Converts the frame to HSV, applies `inRange(lower, upper)` (two bands for red).
- Shows **Original**, **Mask**, and **Filtered** views.
- Press **`c`** to print an easily copy‑pasteable dictionary snippet for your constants.

**Typical outputs.**
```text
# Example
"Blue":   {"Lower": (90, 70, 70), "Upper": (130, 255, 255)}
"Green":  {"Lower": (40, 70, 70), "Upper": (85, 255, 255)}
"Yellow": {"Lower": (20, 100, 100), "Upper": (30, 255, 255)}
# Red uses two bands:
"Red1": {"Lower": (0, 100, 100),   "Upper": (10, 255, 255)}
"Red2": {"Lower": (160, 100, 100), "Upper": (179, 255, 255)}
```

**When to use.** After pre‑processing is tuned. Make sure you test with the **real spheres** and **actual lighting** you’ll race in.

---

### 4) `select_sphero_color_range.py`
**Purpose.** **Mouse‑driven** HSV calibration: draw a **circle** around the sphere; the script extracts pixels inside the circle and computes **5th–95th percentiles** per HSV channel to avoid outliers.

**Outputs.**
```text
Computed HSV range (5th–95th percentile):
lower = [H_low, S_low, V_low]
upper = [H_high, S_high, V_high]
```
Use this as a starting range, then fine‑tune with the `color_trackbar_*.py` tools.

**When to use.** Fast first‑pass calibration when you already have a clean single frame of the sphere.

---

### 5) `select_sphero_size.py`
**Purpose.** Measure **center (x, y)**, **radius**, and **diameter** (in pixels) of your Sphero in the current camera view by **click‑dragging** a circle.

**Outputs.**
```text
Sphere parameters:
center_x = <int>
center_y = <int>
radius   = <int>  # in pixels
diameter = <int>  # in pixels
```
Feed the **radius/diameter** into your tracker (min/max contour area, expected circle radius, etc.).

---

### 6) `path_camera_perspective.py`
**Purpose.** Define a **four‑corner perspective crop** for the **Path** camera. Visualizes: raw frame with labeled corners, cropped canvas (ROI overlay), and **warped top‑down** view.

**Outputs.**
- Printed **Perspective Points** (TL, TR, BL, BR).
- Suggested **Area bounds** and `width × height` of the warp.
- Use these to compute and cache `cv.getPerspectiveTransform()` in your app.

**When to use.** Once the physical camera is mounted and the full path area is visible; ensures consistent coordinates for path drawing and localization.

---

### 7) `finishline_camera_perspective.py`
**Purpose.** Similar to the Path tool, but also lets you position **Start** and **Stop** horizontal lines **in the warped space** for lap timing. Visualizes raw frame, cropped canvas, and **warped** view with lines.

**Outputs.**
- Perspective **corner points** (TL, TR, BL, BR).
- **Start line** `y` & width segment, **Stop line** `y` & width segment in the warped frame.
- Copy these into your detector/timer config to ensure upward crossings (or your chosen direction) are measured correctly.

**When to use.** After you fix the finishline camera position and height; finalizes timing geometry.

---

## Keyboard & UI

- **Drag trackbars** to adjust parameters.
- **`c`** — print **current calibration values** to console (most tools).  
- **`r`** — reset selection (selection tools).  
- **`ESC`** — exit.
- Multiple **named windows** are opened; arrange them on a second display if possible.

---

## Tips & Best Practices

- **Backend choice matters.** On Windows, prefer `cv.CAP_DSHOW` for Logitech C920/C270; on macOS `cv.CAP_AVFOUNDATION`; on Linux `cv.CAP_V4L2`/`cv.CAP_GSTREAMER`.
- **Warm‑up cameras** (read/discard ~10 frames) to stabilize auto‑exposure/white balance before calibration.
- Keep **lighting constant**; if lighting changes (e.g., sunlight), re‑tune **CLAHE/morphology** and **HSV**.
- Cache and **reuse** the perspective matrix; don’t recompute every frame.
- Red often needs **two HSV bands** due to hue wrap‑around.
- Save your printed configs into versioned files (`backend/constants.py`) and commit them per location (lab vs gym).

---

## Outputs to copy into the main app

- **Pre‑processing**: bilateral/median/CLAHE/morphology numbers.  
- **HSV ranges** per color (lower/upper; red has two bands).  
- **Sphere size** in pixels (radius/diameter, center ref if needed).  
- **Perspective corners** and **start/stop** line coordinates.

---

## File List

```
tools/
├─ __init__.py                         # package marker
├─ camera_configuration.py             # backends, CAP_PROP inspection, writability tests
├─ frame_manipulation_parameters.py    # bilateral/median/CLAHE/morphology tuner + masks/contours
├─ color_trackbar_red.py               # HSV trackbars for Red (two hue bands)
├─ color_trackbar_yellow.py            # HSV trackbars for Yellow
├─ color_trackbar_green.py             # HSV trackbars for Green
├─ color_trackbar_blue.py              # HSV trackbars for Blue
├─ select_sphero_color_range.py        # mouse circle → percentile HSV range
├─ select_sphero_size.py               # mouse circle → center/radius/diameter (px)
├─ path_camera_perspective.py          # 4‑corner ROI + perspective warp (Path camera)
└─ finishline_camera_perspective.py    # 4‑corner ROI + warp + start/stop lines (Finishline)
```


