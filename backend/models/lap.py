import uuid
import time
import numpy as np
import cv2 as cv
from datetime import datetime
from backend.models.sphero_bolt import SpheroBold
from backend.constants import PATH, COLOR_RANGES_STRICT, COLOR_RANGES, COLORS_HSV, COLORS_BGR


class Lap:
    def __init__(self, sphere:SpheroBold):
        pass