

HSV_RANGES_STRICT = {
    "Red1"  : {"Lower" : (0, 170, 120),   "Upper" : (10, 255, 255)}, 
    "Red2"  : {"Lower" : (170, 170, 120), "Upper" : (179, 255, 255)}, 
    "Yellow": {"Lower" : (22, 150, 150),  "Upper" : (35, 255, 255)},
    "Green" : {"Lower" : (45, 150, 130),  "Upper" : (75, 255, 255)}, 
    "Blue"  : {"Lower" : (100, 150, 120), "Upper" : (125, 255, 255)},  
 
}

HSV_RANGES_WIDE = {
    "Red1"  : {"Lower" :(0, 50, 50),   "Upper" : (10, 255, 255)},
    "Red2"  : {"Lower" :(170, 50, 50), "Upper" : (179, 255, 255)},
    "Yellow": {"Lower" :(15, 50, 50),  "Upper" : (35, 255, 255)},
    "Green" : {"Lower" :(35, 40, 40),  "Upper" : (85, 255, 255)},
    "Blue"  : {"Lower" :(85, 40, 40),  "Upper" : (135, 255, 255)},
    "Purple": {"Lower" :(135, 40, 40), "Upper" : (160, 255, 255)},
}

HSV_RANGES_NORMAL = {
    "Red1"  : {"Lower" :(0, 100, 100),   "Upper" : (10, 255, 255)},
    "Red2"  : {"Lower" :(160, 100, 100), "Upper" : (180, 255, 255)},
    "Yellow": {"Lower" :(20, 100, 100),  "Upper" : (30, 255, 255)},
    "Green" : {"Lower" :(40, 70, 70),    "Upper" : (85, 255, 255)},
    "Blue"  : {"Lower" :(90, 70, 70),    "Upper" : (130, 255, 255)},
    "Purple": {"Lower" :(130, 50, 50),   "Upper" : (160, 255, 255)},
}




HSV_RANGES_MANUAL = {

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
    "Orange" : (0, 165, 255),
    "Cyan"   : (255, 255, 0),
    "Magenta": (255, 0, 128),
    "Pink"   : (203, 192, 255),
    "White"  : (255, 255, 255),
    "Gray"   : (128, 128, 128),
    "Black"  : (0, 0, 0)  
}



"""
HSV = Tuple[int, int, int]



@dataclass(frozen=True)
class HsvRange:
    LOWER: HSV
    UPPER: HSV



@dataclass(frozen=True)
class HsvRanges:
    RED1   : HsvRange
    RED2   : HsvRange
    YELLOW : HsvRange
    GREEN  : HsvRange
    BLUE   : HsvRange



class HsvColorsRange(Enum):
    NORMAL = HsvRanges(
        RED1   = HsvRange((0, 100, 100),   (10, 255, 255)),
        RED2   = HsvRange((160, 100, 100), (180, 255, 255)),
        YELLOW = HsvRange((20, 100, 100), (30, 255, 255)),
        GREEN  = HsvRange((40, 70, 70),    (85, 255, 255)),
        BLUE   = HsvRange((90, 70, 70),     (130, 255, 255))
    )

    WIDE = HsvRanges(
        RED1   = HsvRange((0, 50, 50),   (10, 255, 255)),
        RED2   = HsvRange((170, 50, 50), (179, 255, 255)),
        YELLOW = HsvRange((15, 50, 50),  (35, 255, 255)),
        GREEN  = HsvRange((35, 40, 40),  (85, 255, 255)),
        BLUE   = HsvRange((85, 40, 40),  (135, 255, 255))
    )

    STRICT = HsvRanges(
        RED1   = HsvRange((0, 170, 120),   (10, 255, 255)),
        RED2   = HsvRange((170, 170, 120), (179, 255, 255)),
        YELLOW = HsvRange((22, 150, 150), (35, 255, 255)),
        GREEN  = HsvRange((45, 150, 130), (75, 255, 255)),
        BLUE   = HsvRange((100, 150, 120), (125, 255, 255))
    )

    MANUAL = HsvRanges(
        RED1   = HsvRange((0, 0, 0), (0, 0, 0)),
        RED2   = HsvRange((0, 0, 0), (0, 0, 0)),
        YELLOW = HsvRange((0, 0, 0), (0, 0, 0)),
        GREEN  = HsvRange((0, 0, 0), (0, 0, 0)),
        BLUE   = HsvRange((0, 0, 0), (0, 0, 0)),
    )
"""