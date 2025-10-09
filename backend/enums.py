import cv2 as cv
from enum import Enum
from backend.constants import HSV_RANGES_MANUAL, HSV_RANGES_NORMAL, HSV_RANGES_STRICT, HSV_RANGES_WIDE



class CaptureApi(Enum):
    Windows = cv.CAP_DSHOW
    Linux = cv.CAP_V4L2
    Mac = cv.CAP_AVFOUNDATION
    FFMPEG = cv.CAP_FFMPEG
    GSTREAMER = cv.CAP_GSTREAMER



class HsvColorsRange(Enum):
    NORMAL = HSV_RANGES_NORMAL
    WIDE = HSV_RANGES_WIDE
    STRICT = HSV_RANGES_STRICT
    MANUAL = HSV_RANGES_MANUAL




class SpheroColor(Enum):
    RED = "Red"
    BLUE = "Blue"
    GREEN = "Green"
    YELLOW = "Yellow"
