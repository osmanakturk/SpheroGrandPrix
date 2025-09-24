# SpheroGrandPrix — BOLT Racing & Vision System
## Rule-Based and HSV-Color-Based Classical CV Segmentation and Tracking System

<p align="center"> 
  <a href="https://git-scm.com/" target="_blank" rel="noreferrer"> 
 <img src="https://www.vectorlogo.zone/logos/git-scm/git-scm-icon.svg" alt="git" width="auto" height="40"/> 
 </a> 
 
 <a href="https://www.w3.org/html/" target="_blank" rel="noreferrer"> 
 <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/html5/html5-original-wordmark.svg" alt="html5" width="40" height="40"/> 
 </a> 
 
 <a href="https://getbootstrap.com" target="_blank" rel="noreferrer"> 
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/bootstrap/bootstrap-plain-wordmark.svg" alt="bootstrap" width="40" height="40"/> 
 </a> 
 
  <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"> 
 <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/javascript/javascript-original.svg" alt="javascript" width="40" height="40"/> 
 </a> 

  <a href="https://www.sqlite.org/" target="_blank" rel="noreferrer"> 
 <img src="https://www.vectorlogo.zone/logos/sqlite/sqlite-icon.svg" alt="sqlite" width="40" height="40"/> 
 </a> 
 
 <a href="https://www.python.org" target="_blank" rel="noreferrer"> 
 <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" width="40" height="40"/> 
 </a> 

 <a href="https://opencv.org/" target="_blank" rel="noreferrer"> 
 <img src="https://www.vectorlogo.zone/logos/opencv/opencv-icon.svg" alt="opencv" width="40" height="40"/> 
 </a> 
 
 <a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer">
 <img src="https://flask.palletsprojects.com/en/stable/_images/flask-name.svg" alt="flask" width="100" height="40"/> 
 </a> 

 </p>


> **Versions**  
> This repository contains **two major versions** of the project.

> • **Version 1** (this branch): Uses **Finishline** camera for precise lap timing (4-point perspective warp + start/stop horizontal lines in warped space) and **Path camera** for live trajectory drawing/localization (top-down warp; per-color centers tracked and written to path canvases).
 
> • **Version 2** (branch version2): Keeps the **Finishline** camera for timing, and introduces a **Status** camera with three polygonal zones **(Back/Middle/Front)** in a single FOV to drive UI cues, warnings, and adaptive rules.

> • **Difference:**

> *v1* = **Path + Finishline** (trajectory + timing).

> *v2* = **Status + Finishline** (zones + timing), replacing the dedicated Path camera with zone-based logic.
---


<div align="center">


<!-- Replace these placeholders with real images in your repo -->
<img src="docs/images/cover.jpg" alt="SpheroGrandPrix Cover" width="70%"/>

<br/><br/>

<img src="docs/images/architecture.png" alt="System Architecture" width="70%"/>

<br/><br/>

<img src="docs/images/game_ui.png" alt="Game UI & Stats" width="70%"/>

</div>

---

## Table of Contents (EN)
- [Overview]
- [What this project achieves]
- [Architecture]
- [Key components]
- [Setup & Run]
- [Runtime Controls]
- [API Endpoints]
- [Calibration workflow (tools/)]
- [Data & Persistence]
- [File structure (v1)]
- [Roadmap]
- [License]

## Inhoudstafel (NL)
- [Overzicht]
- [Wat dit project bereikt]
- [Architectuur]
- [Belangrijkste componenten]
- [Installatie & Starten]
- [Runtime-bediening]
- [API-Eindpunten]
- [Kalibratie workflow (tools/)]
- [Data & Persistentie]
- [Mappenstructuur (v1)]
- [Roadmap]
- [Licentie]

---

## Overview (EN) 
This project is a classical computer vision system that performs **object segmentation and tracking** based on **predefined HSV color ranges** and **rule-based logic**.

**SpheroGrandPrix** is a computer‑vision–driven racing platform for **Sphero BOLT** robots. It captures two camera feeds (**Path** and **Finishline**), runs color‑based detection, draws trajectories, and times laps. The project includes a **web UI** (Bootstrap + Chart.js) for starting/stopping heats, showing live overlays, and viewing results.

- The **`backend/`** contains models/services for cameras, vision, and lap management.
- The **`frontend/`** serves a Bootstrap UI and polling JS for stats and results.
- The **`tools/`** folder provides interactive OpenCV utilities to **calibrate HSV, perspective, pre‑processing, and sphere size**; the app consumes these settings via `backend/configs.py` and `backend/constants.py`.

**Capture backends:** platform‑specific OpenCV backends through an enum (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer/FFMPEG).

---

## What this project achieves (EN) 

- **Reliable, tunable detection** of Sphero BOLTs by color (HSV), with **pre‑processing** (bilateral, median, CLAHE, morphology) to adapt to lighting and noise.
- **Perspective‑corrected views** for path and finishline cameras; **start/finish** line overlays for accurate timing.
- **Per‑color trajectories** and lap timing for up to four spheres (Red/Yellow/Blue/Green), with **PNG exports** that blend with an optional `paths/background.png`.
- **SQLite persistence** of lap results.
- **Dashboard & charts** for quick summaries (doughnut chart), and modal previews of saved path images via the web UI.

---

## Architecture (EN) 

**High‑level flow:**
1. **Camera services** open devices with a chosen backend/resolution; frames are **warped** using a perspective matrix built from user‑defined corner points. Start/finish lines can be drawn in warped space.
2. A **Lap** session aggregates four `SpheroBolt` instances (Red/Yellow/Blue/Green). Each Sphero runs detection on **Path** and **Finishline** frames using configured **HSV ranges** and **pre‑processing** parameters.
3. Detected contours/centers update **per‑color canvases** and timing state; when a race ends, PNG path images are written to `paths/` and DB rows are inserted.
4. The **frontend** polls backend endpoints for lap state and aggregate stats; it renders **Bootstrap** UI components and **Chart.js** doughnut charts, and opens **modals** with saved path images.

**HSV ranges:** Four presets are common: **Normal**, **Wide**, **Strict**, and **Manual**. Red hue is typically split into **two bands** to handle hue wrap‑around.

---

## Key components (EN) 

### Backend
- **`backend/models/camera.py`** — wraps OpenCV capture, manages size/FPS/backend, **perspective transform** (`cv.getPerspectiveTransform`), and optional **start/finish line** overlays in warped space.
- **`backend/models/lap.py`** — orchestrates per‑color `SpheroBolt`s, tracks session state, merges detection outputs, and **saves results to SQLite**.
- **`backend/models/sphero_bolt.py`** — per‑color state (centers/radii/canvases), delegates to the detector, and **saves a path PNG** with a caption (user & lap time).
- **`backend/detectors/detector.py`** — color masking, contour selection, center/radius estimation, and per‑frame updates.
- **`backend/configs.py`** — dataclasses:
  - `CameraConfig`: device/backend + `perspective_points`, optional `start_line`/`finish_line`.
  - `DetectorConfig`: pre‑processing, radius bounds, HSV preset.
  - `SpheroConfig`: initial frames/background & debug flags.
- **`backend/enums.py`** — capture backends & HSV presets.
- **`backend/constants.py`** — HSV ranges and color maps (BGR/HSV).
- **`backend/services/camera_tracker.py`** — loop to open/read cameras, apply perspective, blend overlays, and bridge to `Lap`. Loads optional background underlay from `paths/background.png`.

### Frontend
- **`frontend/templates/*.html`** — Bootstrap pages (home, game, settings).
- **`frontend/static/js/game.js`** — fetches endpoints (start/stop/state/stats, username changes, resets), renders **Chart.js** doughnut, manages modals.
- **`frontend/static/css/bootstrap.css`** and assets — UI styling and logos.

---


## Setup & Run (EN) 

```bash
# 1) Clone the repo
git clone https://github.com/osmanakturk/SpheroGrandPrix.git
cd SpheroGrandPrix

# 2) (Optional) Create & activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Add a background for path exports
# paths/background.png

# 5) Run
python app.py
# Open: http://localhost:5000

```

**Cameras:** set `CameraConfig` with `cap_index` (or `cap_source`) and a platform backend (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer). Define `perspective_points` and optional `start_line` / `finish_line` for the finish camera.

---

## Runtime Controls (EN)


- **Start Lap** → begins timing and tracking for all colors.
- **Stop Lap** → finalizes timing, computes per-color results and aggregate scores.
- **Reset <color>** → clears state for a single color (red, yellow, blue, green).
- **Change Username** → sets display name per color (shown in UI and results).
- **Maintenance (optional UI actions):**
  - *Release Cameras:* frees all VideoCapture handles
  - *Delete Database:* removes database.sqlite and restarts with empty results
- **Live Streams:**
  - *Path camera:* top-down trajectory visualization.
  - *Finishline camera:* rectified view with Start/Stop lines for crossings.

## API Endpoints (EN)

> - `Base URL:` http://<host>:5000
> - `GET /video_feed/path` — Stream — MJPEG feed from the Path camera (top-down/warped view).  
> - `GET /video_feed/finishline` — Stream — MJPEG feed from the Finishline camera (warped with timing lines).  
> - `POST /lap/start` — Action — start a new lap for all colors.  
> - `POST /lap/stop` — Action — stop the lap, compute results and scores.  
> - `GET /lap/state` — Status — current per-color state.  
> - `GET /stats` — Aggregates — totals and dashboard entries.  
> - `POST /reset/<color>` — Action — reset state for a single color (red|yellow|blue|green).  
> - `POST /username_change/<color>/<username>` — Action — set display name for a color.
> - `GET /paths/<filename>` — Static — serve saved path/score images (as listed in /stats.dashboard[].img_path).
> - `GET /release_cap` — Operational — release all open cameras.
> - `GET /delete/database` — Operational — delete database.sqlite (if present).

---


## Calibration workflow (tools/) (EN) 

1. **Inspect camera & backends** → `tools/camera_configuration.py`  
2. **Tune pre‑processing** (bilateral/median/CLAHE/morphology) → `tools/frame_manipulation_parameters.py`  
3. **Calibrate HSV**  
   - Live trackbars per color → `tools/color_trackbar_red|yellow|green|blue.py`  
   - Percentile HSV from circular ROI → `tools/select_sphero_color_range.py`  
4. **Measure sphere size** (px radius/diameter) → `tools/select_sphero_size.py`  
5. **Perspective crop**  
   - Path ROI → `tools/path_camera_perspective.py`  
   - Finishline ROI + start/stop lines → `tools/finishline_camera_perspective.py`

> Tip: Save final numbers into config (e.g., `backend/constants.py`, `backend/configs.py`) per venue/lighting.

---

## Data & Persistence (EN) 

- **SQLite file:** `database.sqlite`  
  Table example: `sphero_bolt(id, lap_id, color, username, path_img_path, start_time, finish_time, total_lap_time)`
- **Path images:** `paths/<lap>_<COLOR>.png` contain the per‑color canvas (optionally blended with `paths/background.png`) and a caption bar with username + lap time.

---

## File structure (v1) (EN) 

```
SpheroGrandPrix/
├─ backend/
│  ├─ configs.py            # CameraConfig, DetectorConfig, SpheroConfig (dataclasses)
│  ├─ constants.py          # HSV presets + BGR/HSV maps
│  ├─ enums.py              # capture APIs & HSV presets
│  ├─ detectors/
│  │  └─ detector.py        # color/contour detection logic
│  ├─ models/
│  │  ├─ camera.py          # capture, perspective warp, start/finish overlays
│  │  ├─ lap.py             # session orchestration, SQLite save
│  │  └─ sphero_bolt.py     # per-color detection state, path image export
│  └─ services/
│     └─ camera_tracker.py  # open/read cameras, perspective, glue to Lap
├─ frontend/
│  ├─ templates/            # Bootstrap pages (home, game, settings)
│  └─ static/               # js (game.js), css (bootstrap.css), assets (logos, images)
├─ tools/                   # OpenCV calibration UIs (HSV, perspective, sizing, pre-processing)
├─ paths/                   # background.png + saved path PNGs
├─ tests/                   # demo videos, Bluetooth test
├─ app.py                   # web server & endpoints
└─ database.sqlite          # results
```

---

## Roadmap (EN) 
- **Branch `version2`:** status camera zones, extended tooling, refactors.
- Robust camera reconnection & FPS scaling.
- Optional GPU/OpenCL toggles and async pipelines.

---

## License 
MIT 

---


# Overzicht (NL) 

Dit project is een klassiek computer vision-systeem dat **objectsegmentatie en tracking** uitvoert op basis van **vooraf gedefinieerde HSV-kleurbereiken** en **regelgebaseerde logica**.

**SpheroGrandPrix** is een vision‑gebaseerd raceplatform voor **Sphero BOLT** robots. Twee camera’s (**Pad** en **Finish**) worden gecorrigeerd naar bovenaanzicht, waarna kleurdetectie, trajecttekenen en rondetiming gebeuren. Een **web‑UI** (Bootstrap + Chart.js) laat je heats starten/stoppen en resultaten bekijken.

- **`backend/`**: modellen & services voor camera’s, detectie en rondes.
- **`frontend/`**: Bootstrap UI + JavaScript voor polling en grafieken.
- **`tools/`**: interactieve OpenCV‑hulpen voor **HSV‑kalibratie, perspectief** en **voorbewerking**.

**Capture backends:** platform‑specifiek via een enum (Windows/DirectShow, Linux/V4L2/GStreamer/FFMPEG, macOS/AVFoundation).

---

## Wat dit project bereikt (NL) 

- **Betrouwbare detectie** per kleur (HSV) met **instelbare voorbewerking** (bilateral, median, CLAHE, morfologie).
- **Perspectiefcorrectie** voor pad‑ en finish‑camera’s; **start/finish‑lijnen** in het gewarpte beeld.
- **Kleursporen** en rondetijden voor vier bollen (Rood/Geel/Blauw/Groen), plus **PNG‑exports** met achtergrond.
- **SQLite‑opslag** van resultaten.
- **Dashboard & grafieken** (doughnut), modals met opgeslagen pad‑afbeeldingen.

---

## Architectuur (NL) 

**Flow:**
1. **Camera‑service** opent het device met gevraagde backend en resolutie; frames worden **gewarpt** via een perspectiefmatrix op basis van hoekpunten. Start/finish‑lijnen kunnen in het gewarpte beeld getekend worden.
2. Een **Lap** sessie groepeert vier `SpheroBolt`‑instanties. Detectie gebruikt **HSV‑presets** en **voorbewerking**.
3. Gevonden contouren/centra updaten **per‑kleur canvassen** en timing; bij stop worden **PNG’s** geschreven en DB‑rijen ingevoegd.
4. **Frontend** pollt endpoints voor status/statistiek, rendert **Bootstrap** UI en **Chart.js** grafieken, en toont **modals** met pad‑afbeeldingen.

**HSV‑reeksen:** Vier presets: **Normal**, **Wide**, **Strict**, **Manual**. Rood gebruikt **twee banden** wegens hue wrap‑around.

---

## Belangrijkste componenten (NL) 

- **`backend/models/camera.py`** — capture wrapper met **perspectief** en **start/finish‑overlay** in gewarpte ruimte.  
- **`backend/models/lap.py`** — sessiebeheer, merge van detectie, **SQLite opslaan**.  
- **`backend/models/sphero_bolt.py`** — per‑kleur status en **pad‑PNG export** met koptekst.  
- **`backend/detectors/detector.py`** — kleurmaskers, contourselectie, center/radius‑schatting.  
- **`backend/configs.py`** — `CameraConfig`, `DetectorConfig`, `SpheroConfig`.  
- **`backend/enums.py`** — capture backends & HSV‑presets.  
- **`backend/constants.py`** — HSV‑presets + BGR/HSV kleurtabellen.  
- **`backend/services/camera_tracker.py`** — open/lezen camera’s, perspectief, koppeling naar `Lap`, en optionele **achtergrond** (`paths/background.png`).  
- **`frontend/static/js/game.js`** — UI‑polling, endpoints, **Chart.js** doughnut.

---

## Installatie & Starten (NL) 

```bash
# 1) Clone the repo
git clone https://github.com/osmanakturk/SpheroGrandPrix.git
cd SpheroGrandPrix

# 2) (Optional) Create & activate a virtualenv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Add a background for path exports
# paths/background.png

# 5) Run
python app.py
# Open: http://localhost:5000
```

**Camera’s:** stel `CameraConfig` in met `cap_index` of `cap_source` en kies een backend (Windows → DirectShow, macOS → AVFoundation, Linux → V4L2/GStreamer). Definieer `perspective_points` en (indien nodig) `start_line`/`finish_line` voor de finish‑camera.

---

## Runtime-bediening (NL)


- **Start Lap** → start timing & tracking voor alle kleuren..
- **Stop Lap** → rond af, bereken resultaten en totaalscores.
- **Reset <color>** → reset status voor één kleur (red, yellow, blue, green).
- **Change Username** → stel weergavenaam per kleur in.
- **Onderhoud (optionele UI-acties):**
  - *Release Cameras:* geeft alle VideoCapture-handles vrij
  - *Delete Database:* verwijdert database.sqlite en start met lege resultaten
- **Live Streams:**
  - *Path camera:* top-down traject-visualisatie
  - *Finishline camera:* gerectificeerd beeld met Start/Stop lijnen.

## API-Eindpunten (NL)

> - `Base URL:` http://<host>:5000
> - `GET /video_feed/path` — Stream — MJPEG feed van de Path camera (top-down/warp). 
> - `GET /video_feed/finishline` — Stream — MJPEG feed van de Finishline camera (warp met timelijnen).  
> - `POST /lap/start` — Actie — start een nieuwe ronde (alle kleuren). 
> - `POST /lap/stop` — Actie — stop de ronde, bereken resultaten en scores.
> - `GET /lap/state` — Status — actuele status per kleur.
> - `GET /stats` — Totalen — aggregaten en dashboard-items.  
> - `POST /reset/<color>` — Actie — reset voor één kleur (red|yellow|blue|green). 
> - `POST /username_change/<color>/<username>` — Actie — stel weergavenaam in.
> - `GET /paths/<filename>` — Static — serveer opgeslagen pad/score-afbeeldingen (zoals in /stats.dashboard[].img_path).
> - `GET /release_cap` — Operationeel — geef alle camera’s vrij.
> - `GET /delete/database` — Operationeel — verwijder database.sqlite (indien aanwezig).

---


## Kalibratie workflow (tools/) (NL) 

1. **Cameraproperties/backends** → `tools/camera_configuration.py`  
2. **Voorbewerking tunen** → `tools/frame_manipulation_parameters.py`  
3. **HSV kalibratie**  
   - Live trackbars per kleur → `tools/color_trackbar_*.py`  
   - Percentiel‑HSV via cirkel‑ROI → `tools/select_sphero_color_range.py`  
4. **Sferediameter (px)** → `tools/select_sphero_size.py`  
5. **Perspectief**  
   - Pad‑ROI → `tools/path_camera_perspective.py`  
   - Finish‑ROI + start/stop → `tools/finishline_camera_perspective.py`  

---

## Data & Persistentie (NL) 

- **SQLite:** `database.sqlite` met tabel `sphero_bolt` (id, lap_id, color, username, path_img_path, start_time, finish_time, total_lap_time).  
- **Pad‑afbeeldingen:** `paths/<lap>_<KLEUR>.png` met optionele blend van `paths/background.png`.

---

## Mappenstructuur (v1) (NL) 

```
SpheroGrandPrix/
├─ backend/ (configs, constants, enums, detectors, models, services)
├─ frontend/ (templates, static/js, css, assets)
├─ tools/    (OpenCV kalibratiehulpen)
├─ paths/    (background.png + PNG-exports)
├─ tests/    (demovideo's, Bluetooth test)
├─ app.py
└─ database.sqlite
```

---

## Roadmap (NL) 
- **Branch `version2`:** uitgebreidere zone‑logica, extra tooling en refactors.
- Robuustere camera‑reconnect & FPS‑optimalisaties.
- Optionele GPU/OpenCL & asynchrone pipelines.

---

## License / Licentie
MIT
