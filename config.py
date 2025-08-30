import cv2 as cv
import numpy as np
from pathlib import Path


PATH = Path(__file__).resolve()



COLOR_RANGES_STRICT = {
    "Red1"  : {"Lower" : (0, 170, 120),   "Upper" : (10, 255, 255)}, 
    "Red2"  : {"Lower" : (170, 170, 120), "Upper" : (179, 255, 255)}, 
    "Yellow": {"Lower" : (22, 150, 150),  "Upper" : (35, 255, 255)},
    "Green" : {"Lower" : (45, 150, 130),  "Upper" : (75, 255, 255)}, 
    "Blue"  : {"Lower" : (100, 150, 120), "Upper" : (125, 255, 255)},  
    "Purple": {"Lower" : (138, 140, 120), "Upper" : (158, 255, 255)}    
}

COLOR_RANGES = {
    "Red1"  : {"Lower" :(0, 50, 50),   "Upper" : (10, 255, 255)},
    "Red2"  : {"Lower" :(170, 50, 50), "Upper" : (179, 255, 255)},
    "Yellow": {"Lower" :(15, 50, 50),  "Upper" : (35, 255, 255)},
    "Green" : {"Lower" :(35, 40, 40),  "Upper" : (85, 255, 255)},
    "Blue"  : {"Lower" :(85, 40, 40),  "Upper" : (135, 255, 255)},
    "Purple": {"Lower" :(135, 40, 40), "Upper" : (160, 255, 255)}
}

COLORS_HSV = {
    "Red"   : (0, 255, 255),
    "Yellow": (30, 255, 255),
    "Green" : (60, 255, 255),
    "Blue"  : (120, 255, 255),
    "Purple": (150, 255, 255)    
}


COLORS_BGR= {
    "Red"    : (0, 0, 255),
    "Yellow" : (0, 255, 255),
    "Green"  : (0, 255, 0),
    "Blue"   : (255, 0, 0),
    "Purple" : (255, 0, 255),
    "Purple" : (255, 0, 255),
    "Orange" : (0, 165, 255),
    "Cyan"   : (255, 255, 0),
    "Magenta": (255, 0, 128),
    "Pink"   : (203, 192, 255),
    "White"  : (255, 255, 255),
    "Gray"   : (128, 128, 128),
    "Black"  : (0, 0, 0)  
}

