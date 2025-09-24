# SpheroGrandPrix — BOLT Racing & Vision System
## Rule-Based and HSV-Color-Based Classical CV Segmentation and Tracking System

<p align="center">
  <a href="https://git-scm.com/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" height="40"/>
  </a>
  <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" height="40"/>
  </a>
  <a href="https://getbootstrap.com" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" height="40"/>
  </a>
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" height="40"/>
  </a>
  <a href="https://www.sqlite.org/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/sqlite/sqlite-icon.svg" alt="sqlite" height="40"/>
  </a>
  <a href="https://www.python.org" target="_blank" rel="noreferrer">
    <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" height="40"/>
  </a>
  <a href="https://opencv.org/" target="_blank" rel="noreferrer">
    <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" height="40"/>
  </a>
  <a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer">
    <img src="https://flask.palletsprojects.com/en/stable/_images/flask-name.svg" alt="flask" height="40"/>
  </a>
</p>

> **Versions**  
> This repository contains **two major versions** of the project.  
>
> • **Version 2** (this branch, **`version2`**): Uses the **Finishline** camera for precise lap timing (4-point perspective warp + start/stop horizontal lines in warped space) and a **Status** camera that defines three polygonal zones (**Back/Middle/Front**) within a single FOV to drive UI cues, warnings, and adaptive rules.  
>
> • **Version 1** (branch **`main`**): Uses **Finishline** for timing and a separate **Path** camera for top-down trajectory drawing/localization.  
>
> • **Difference:**  
> *v1* = **Path + Finishline** (trajectory + timing).  
> *v2* = **Status + Finishline** (zones + timing), replacing the dedicated Path camera with zone-based logic.

---

<div align="center">

<!-- Replace these placeholders with real images in your repo -->
<img src="docs/images/cover_v2.jpg" alt="SpheroGrandPrix v2 Cover" width="70%"/>

<br/><br/>

<img src="docs/images/architecture_v2.png" alt="System Architecture v2" width="70%"/>

<br/><br/>

<img src="docs/images/game_ui_v2.png" alt="Game UI & Stats v2" width="70%"/>

</div>

---

## Table of Contents (EN)
- [Overview](#overview-en)
- [What this project achieves](#what-this-project-achieves-en)
- [Architecture](#architecture-en)
- [Key components](#key-components-en)
- [Setup & Run](#setup--run-en)
- [Runtime Controls](#runtime-controls-en)
- [API Endpoints](#api-endpoints-en)
- [Calibration workflow (tools/)](#calibration-workflow-tools-en)
- [Data & Persistence](#data--persistence-en)
- [File structure (v2)](#file-structure-v2-en)
- [Roadmap](#roadmap-en)
- [License](#license--licentie)

## Inhoudstafel (NL)
- [Overzicht](#overzicht-nl)
- [Wat dit project bereikt](#wat-dit-project-bereikt-nl)
- [Architectuur](#architectuur-nl)
- [Belangrijkste componenten](#belangrijkste-componenten-nl)
- [Installatie & Starten](#installatie--starten-nl)
- [Runtime-bediening](#runtime-bediening-nl)
- [API-Eindpunten](#api-eindpunten-nl)
- [Kalibratie workflow (tools/)](#kalibratie-workflow-tools-nl)
- [Data & Persistentie](#data--persistentie-nl)
- [Mappenstructuur (v2)](#mappenstructuur-v2-nl)
- [Roadmap](#roadmap-nl)
- [Licentie](#license--licentie)

---

## Overview (EN)
This is a classical computer vision system that performs **segmentation and tracking** with **predefined HSV color ranges** and **rule-based logic**.

**SpheroGrandPrix (v2)** is a computer-vision–driven racing platform for **Sphero BOLT** robots. It captures two camera feeds (**Status** and **Finishline**), runs color-based detection, overlays **zone polygons** on the status feed, and times laps using the finishline feed. The project includes a **web UI** (Bootstrap + Chart.js) for starting/stopping heats, showing live overlays, and viewing results.

- The **`backend/`** contains models/services for cameras, vision, zone logic, and lap management.
- The **`frontend/`** serves a Bootstrap UI and polling JS for stats and results.
- The **`tools/`** folder provides interactive OpenCV utilities to **calibrate HSV, pre-processing, finishline perspective, and status zone geometry**; the app consumes these settings via `backend/configs.py` and `backend/constants.py`.

**Capture backends:** platform-specific OpenCV backends through an enum (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer/FFMPEG).

---

## What this project achieves (EN)
- **Reliable, tunable detection** of Sphero BOLTs by color (HSV), with **pre-processing** (bilateral, median, CLAHE, morphology) to adapt to lighting and noise.
- **Zone-aware UX & logic:** a single **Status** camera FOV split into **Back/Middle/Front** polygons for UI cues, warnings, and rule triggers.
- **Accurate lap timing** on a perspective-corrected **Finishline** feed with **start/stop** lines in warped space.
- **Per-color lap results** for up to four spheres (Red/Yellow/Blue/Green) with **SQLite persistence** and dashboard summaries (doughnut chart).
- Optional **snapshots/overlays** saved for debugging or results presentation.

---

## Architecture (EN)
**High-level flow:**
1. **Camera services** open devices with a chosen backend/resolution.
   - **Status camera:** draws **Back/Middle/Front** polygons on the live frame (or on a pre-warped canvas if configured), using coordinates extracted from the calibration tool.
   - **Finishline camera:** applies **perspective warp** and draws **start/stop** horizontal lines in warped space.
2. A **Lap** session aggregates four `SpheroBolt` instances (Red/Yellow/Blue/Green). Each sphere runs color detection on frames and updates **timing state** (crossings on the finishline feed) and **UI hints** (zone presence on the status feed).
3. When a race ends, results are persisted to **SQLite**; optional snapshot images may be written to `paths/` for dashboard use.
4. The **frontend** polls backend endpoints for lap state and aggregate stats; it renders **Bootstrap** UI components and **Chart.js** doughnut charts, and can present **modal previews** of saved images.

**HSV ranges:** Presets like **Normal**, **Wide**, **Strict**, and **Manual**; **Red** usually needs **two hue bands** due to wrap-around.

---

## Key components (EN)

### Backend
- **`backend/models/camera.py`** — wraps OpenCV capture, manages size/FPS/backend.  
  - **Status mode:** renders **zone polygons** (Back/Middle/Front) using calibrated points (A, BL/CL/CR/BR, DL/DR, CM).  
  - **Finishline mode:** performs **perspective transform** (`cv.getPerspectiveTransform`) and draws **start/stop** lines in warped space.
- **`backend/models/lap.py`** — orchestrates per-color `SpheroBolt`s, tracks session state, merges detection outputs, and **saves results to SQLite**.
- **`backend/models/sphero_bolt.py`** — per-color state (centers/radii), delegates to the detector, and contributes to timing & UI state.
- **`backend/detectors/detector.py`** — color masking, contour selection, center/radius estimation, and per-frame updates.
- **`backend/configs.py`** — dataclasses:
  - `CameraConfig` for **Status**/**Finishline** devices and geometry (`status_zones`, `perspective_points`, `start_line`, `finish_line`).
  - `DetectorConfig` for pre-processing, radius bounds, HSV preset.
  - `SpheroConfig` for initial frames & debug flags.
- **`backend/enums.py`** — capture backends & HSV presets.
- **`backend/constants.py`** — HSV ranges and color maps (BGR/HSV).
- **`backend/services/camera_tracker.py`** — opens/reads cameras, blends overlays (zones/lines), and bridges to `Lap`.

### Frontend
- **`frontend/templates/*.html`** — Bootstrap pages (home, game, settings).
- **`frontend/static/js/game.js`** — fetches endpoints (start/stop/state/stats, username changes, resets), renders **Chart.js** doughnut, manages modals.
- **`frontend/static/css/bootstrap.css`** and assets — UI styling and logos.

---

## Setup & Run (EN)

```bash
# 1) Clone the repo (switch to version2 branch)
git clone https://github.com/osmanakturk/SpheroGrandPrix.git
cd SpheroGrandPrix
git checkout version2

# 2) (Optional) Create & activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run
python app.py
# Open: http://localhost:5000
```

**Cameras:** set `CameraConfig` with `cap_index` (or `cap_source`) and a platform backend (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer). For **Finishline**, define `perspective_points` and `start_line`/`finish_line`. For **Status**, define calibrated **zone polygons**.

---

## Runtime Controls (EN)

- **Start Lap** → begins timing for all colors (Finishline crossings enabled).
- **Stop Lap** → finalizes timing, computes per-color results and aggregates.
- **Reset `<color>`** → clears state for a single color (`red`, `yellow`, `blue`, `green`).
- **Change Username** → sets display name per color (shown in UI/results).
- **Live Streams:**
  - *Status camera:* single-FOV view with **Back/Middle/Front** polygon overlays (semi-transparent), labeled anchor points and guide lines.
  - *Finishline camera:* rectified/warped view with **Start/Stop** lines for crossing detection.

## API Endpoints (EN)

- `Base URL:` `http://<host>:5000`
- `GET /video_feed/status` — MJPEG feed from the **Status** camera (zones overlaid).
- `GET /video_feed/finishline` — MJPEG feed from the **Finishline** camera (warped with timing lines).
- `POST /lap/start` — start a new lap for all colors.
- `POST /lap/stop` — stop the lap, compute results and scores.
- `GET /lap/state` — current per-color state.
- `GET /stats` — totals and dashboard entries.
- `POST /reset/<color>` — reset state for a single color (`red|yellow|blue|green`).
- `POST /username_change/<color>/<username>` — set display name for a color.
- `GET /paths/<filename>` — serve saved images (e.g., dashboard thumbnails).

---

## Calibration workflow (tools/) (EN)

1. **Inspect camera & backends** → `tools/camera_configuration.py`  
2. **Tune pre-processing** (bilateral/median/CLAHE/morphology) → `tools/frame_manipulation_parameters.py`  
3. **Calibrate HSV**
   - Live trackbars per color → `tools/color_trackbar_red|yellow|green|blue.py`
   - Percentile HSV from circular ROI → `tools/select_sphero_color_range.py`
4. **Measure sphere size** (px radius/diameter) → `tools/select_sphero_size.py`
5. **Geometry**
   - **Finishline** ROI + start/stop lines → `tools/finishline_camera_perspective.py`
   - **Status** zone polygons (**Back/Middle/Front**) → `tools/set_status_camera_zone.py`

> Tip: Save final numbers into config (e.g., `backend/constants.py`, `backend/configs.py`) per venue/lighting; keep the **apex A** stable between sessions for Status zones.

---

## Data & Persistence (EN)
- **SQLite file:** `database.sqlite`  
  Example table: `sphero_bolt(id, lap_id, color, username, img_path, start_time, finish_time, total_lap_time)`
- **Images (optional):** `paths/` may store **status snapshots** (zones overlaid) or **finishline snapshots** for dashboards and audits.

---

## File structure (v2) (EN)
```
SpheroGrandPrix/
├─ backend/
│  ├─ configs.py            # CameraConfig (status/finishline), DetectorConfig, SpheroConfig
│  ├─ constants.py          # HSV presets + BGR/HSV maps
│  ├─ enums.py              # capture APIs & HSV presets
│  ├─ detectors/
│  │  └─ detector.py        # color/contour detection logic
│  ├─ models/
│  │  ├─ camera.py          # status zone overlay & finishline perspective/lines
│  │  ├─ lap.py             # session orchestration, SQLite save
│  │  └─ sphero_bolt.py     # per-color detection state
│  └─ services/
│     └─ camera_tracker.py  # open/read cameras, overlays, glue to Lap
├─ frontend/
│  ├─ templates/            # Bootstrap pages (home, game, settings)
│  └─ static/               # js (game.js), css (bootstrap.css), assets (logos, images)
├─ tools/                   # calibration UIs (HSV, pre-processing, finishline, status zones)
├─ paths/                   # optional snapshots (status/finishline)
├─ tests/                   # demo videos, Bluetooth test
├─ app.py                   # web server & endpoints
└─ database.sqlite          # results
```

---

## Roadmap (EN)
- Refine zone-based rule engine (per-zone penalties/bonuses).
- Robust camera reconnection & FPS scaling.
- Optional GPU/OpenCL toggles and async pipelines.

---

# Overzicht (NL)
Dit is een klassiek computer-vision-systeem dat **segmentatie en tracking** uitvoert met **vooraf gedefinieerde HSV-kleuren** en **regelgebaseerde logica**.

**SpheroGrandPrix (v2)** gebruikt twee feeds (**Status** en **Finishline**): de statuscamera toont **zone-polygonen** (Achter/Midden/Voor) in één FOV voor UI-signalen en regels, terwijl de finishline-camera een **perspectief-warp** en **start/stop** lijnen gebruikt voor nauwkeurige rondetiming. De **web-UI** (Bootstrap + Chart.js) laat heats starten/stoppen, live overlays zien en resultaten bekijken.

- **`backend/`**: modellen/services voor camera’s, detectie, zone-logica en rondes.
- **`frontend/`**: Bootstrap-UI + JavaScript voor polling, dashboards en grafieken.
- **`tools/`**: OpenCV-hulpen voor **HSV-kalibratie, voorbewerking, finishline-perspectief** en **status-zones**.

**Capture backends:** platform-specifiek via een enum (Windows/DirectShow, Linux/V4L2/GStreamer/FFMPEG, macOS/AVFoundation).

---

## Wat dit project bereikt (NL)
- **Betrouwbare, instelbare detectie** per kleur (HSV) met **voorbewerking** (bilateral, median, CLAHE, morfologie).
- **Zone-bewuste UX & logica:** één **Status**-FOV opgedeeld in **Achter/Midden/Voor** polygonen voor UI-signalen, waarschuwingen en regels.
- **Nauwkeurige rondetiming** op een ge-warpede **Finishline** feed met **start/stop** lijnen.
- **Resultaten per kleur** (Rood/Geel/Blauw/Groen) met **SQLite** opslag en dashboard-overzichten (doughnut).
- Optioneel **snapshots/overlays** voor debugging of rapportage.

---

## Architectuur (NL)
**Flow:**
1. **Cameraservice** opent devices met gevraagde backend/resolutie.  
   - **Statuscamera:** tekent **Achter/Midden/Voor** polygonen (gekalibreerde punten A, BL/CL/CR/BR, DL/DR, CM).  
   - **Finishline:** voert **perspectief-warp** uit en tekent **start/stop** lijnen.
2. Een **Lap** sessie groepeert vier `SpheroBolt`-instanties. Detectie voedt **timing** (finishline-kruisingen) en **UI-signalen** (zone-aanwezigheid).
3. Bij stop worden resultaten opgeslagen in **SQLite**; optioneel worden snapshots bewaard in `paths/`.
4. **Frontend** pollt endpoints, rendert **Bootstrap** en **Chart.js**, en kan **modals** met snapshots tonen.

**HSV-reeksen:** Presets **Normal/Wide/Strict/Manual**; **Rood** gebruikt vaak **twee banden** (hue wrap-around).

---

## Belangrijkste componenten (NL)
- **`backend/models/camera.py`** — capture wrapper.  
  - **Statusmodus:** tekent **zone-polygonen** (Achter/Midden/Voor) o.b.v. kalibratiepunten.  
  - **Finishlinemodus:** **perspectief-warp** + **start/stop** lijnen.
- **`backend/models/lap.py`** — sessiebeheer, merge van detectie, **SQLite** opslaan.
- **`backend/models/sphero_bolt.py`** — per-kleur status en detectie.
- **`backend/detectors/detector.py`** — kleurmaskers, contourselectie, center/radius-schatting.
- **`backend/configs.py`** — `CameraConfig` (status/finishline), `DetectorConfig`, `SpheroConfig`.
- **`backend/enums.py`** — capture backends & HSV-presets.
- **`backend/constants.py`** — HSV-presets + BGR/HSV kleurtabellen.
- **`backend/services/camera_tracker.py`** — open/lezen camera’s, overlays, koppeling naar `Lap`.

### Frontend
- **`frontend/templates/*.html`** — Bootstrap pagina’s (home, game, settings).
- **`frontend/static/js/game.js`** — polling van endpoints, **Chart.js** doughnut, modals.
- **`frontend/static/css/bootstrap.css`** en assets — UI-styling en logo’s.

---

## Installatie & Starten (NL)

```bash
# 1) Repo clonen en naar version2 switchen
git clone https://github.com/osmanakturk/SpheroGrandPrix.git
cd SpheroGrandPrix
git checkout version2

# 2) (Optioneel) Virtuele omgeving
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Dependencies
pip install -r requirements.txt

# 4) Starten
python app.py
# Open: http://localhost:5000
```

**Camera’s:** stel `CameraConfig` in met `cap_index` of `cap_source` en kies backend (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer). Voor **Finishline**: `perspective_points` + `start_line`/`finish_line`. Voor **Status**: gekalibreerde **zone-polygonen**.

---

## Runtime-bediening (NL)
- **Start Lap** → start timing voor alle kleuren (Finishline-kruisingen actief).
- **Stop Lap** → rond af, bereken resultaten en totaalscores.
- **Reset `<color>`** → reset status voor één kleur (`red`, `yellow`, `blue`, `green`).
- **Change Username** → stel weergavenaam per kleur in.
- **Live Streams:**
  - *Statuscamera:* één FOV met **Achter/Midden/Voor** overlays (semi-transparant), gelabelde ankerpunten en hulplijnen.
  - *Finishline camera:* ge-warpede view met **Start/Stop** lijnen.

## API-Eindpunten (NL)
- `Base URL:` `http://<host>:5000`
- `GET /video_feed/status` — MJPEG feed van de **Status**camera (met zones).
- `GET /video_feed/finishline` — MJPEG feed van de **Finishline**camera (warp met timelijnen).
- `POST /lap/start` — start een nieuwe ronde (alle kleuren).
- `POST /lap/stop` — stop de ronde, bereken resultaten en scores.
- `GET /lap/state` — actuele status per kleur.
- `GET /stats` — totalen en dashboard-items.
- `POST /reset/<color>` — reset voor één kleur (`red|yellow|blue|green`).
- `POST /username_change/<color>/<username>` — stel weergavenaam in.
- `GET /paths/<filename>` — serveer opgeslagen afbeeldingen (bijv. dashboard thumbnails).

---

## Kalibratie workflow (tools/) (NL)
1. **Cameraproperties/backends** → `tools/camera_configuration.py`  
2. **Voorbewerking tunen** → `tools/frame_manipulation_parameters.py`  
3. **HSV kalibratie**
   - Live trackbars per kleur → `tools/color_trackbar_*.py`
   - Percentiel-HSV via cirkel-ROI → `tools/select_sphero_color_range.py`
4. **Sferediameter (px)** → `tools/select_sphero_size.py`
5. **Geometrie**
   - **Finishline** ROI + start/stop → `tools/finishline_camera_perspective.py`
   - **Status** zone-polygonen (**Achter/Midden/Voor**) → `tools/set_status_camera_zone.py`

---

## Data & Persistentie (NL)
- **SQLite:** `database.sqlite` met tabel `sphero_bolt` (id, lap_id, color, username, img_path, start_time, finish_time, total_lap_time).  
- **Afbeeldingen (optioneel):** `paths/` kan **status-snapshots** (met zones) of **finishline-snapshots** bevatten voor dashboards en audits.

---

## Mappenstructuur (v2) (NL)
```
SpheroGrandPrix/
├─ backend/ (configs, constants, enums, detectors, models, services)
├─ frontend/ (templates, static/js, css, assets)
├─ tools/    (kalibratie: HSV, pre-processing, finishline, status-zones)
├─ paths/    (optionele snapshots)
├─ tests/    (demovideo's, Bluetooth test)
├─ app.py
└─ database.sqlite
```

---

## Roadmap (NL)
- Zone-regelengine verfijnen (per-zone penalties/bonussen).
- Betere camera-reconnect & FPS-optimalisaties.
- Optionele GPU/OpenCL en asynchrone pipelines.

---

## License / Licentie
MIT
