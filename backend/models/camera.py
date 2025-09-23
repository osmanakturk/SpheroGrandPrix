import cv2 as cv
import numpy as np
from typing import Optional, Tuple
from backend.constants import COLORS_BGR
from backend.enums import CaptureApi
from backend.configs import CameraConfig





class Camera():
    
    def __init__(self, config: CameraConfig):

        self._cap_index = config.cap_index
        self._cap_source = config.cap_source
        self._cap_api = config.cap_api.value if config.cap_api is not None else None
        self._cap_width = config.cap_width or 640
        self._cap_height = config.cap_height or 480
        self._cap_fps = config.cap_fps or 30
        self._start_line = config.start_line
        self._finish_line = config.finish_line
        self._perspective_points = config.perspective_points
        self._perspective_width = config.perspective_width
        self._perspective_height = config.perspective_height
        self._cap: Optional[cv.VideoCapture] = None
        self._frame: Optional[cv.typing.MatLike] = None
        self._perspective_frame: Optional[cv.typing.MatLike] = None
        self._perspective_matrix: Optional[cv.typing.MatLike] = None




    @property
    def cap(self) -> Optional[cv.VideoCapture]:
        return self._cap

    @cap.setter
    def cap(self, cap: cv.VideoCapture) -> None:
        self._cap = cap


    @property
    def frame(self) -> Optional[cv.typing.MatLike]:
        return self._frame
    
    @frame.setter
    def frame(self, frame: cv.typing.MatLike) -> None:
        self._frame = frame


    @property
    def cap_index(self) -> Optional[int]:
        return self._cap_index
    

    @cap_index.setter
    def cap_index(self, cap_index: int) -> None:
        self._cap_index = cap_index


    @property
    def cap_source(self) -> Optional[str]:
        return self._cap_source
    
    @cap_source.setter
    def cap_source(self, cap_source: str) ->  None:
        self._cap_source = cap_source

    
    @property
    def cap_api(self) -> int:
        return self._cap_api
    
    @cap_api.setter
    def cap_api(self, cap_api:CaptureApi) -> None:
        self._cap_api = cap_api.value

    @property
    def cap_width(self) -> int:
        return self._cap_width
    
    @cap_width.setter
    def cap_width(self, cap_width: int) -> None:
        self._cap_width = cap_width

    
    @property
    def cap_height(self) -> int:
        return self._cap_height
    
    @cap_height.setter
    def cap_height(self, cap_height: int) -> None:
        self._cap_height = cap_height


    @property
    def cap_fps(self) -> int:
        return self._cap_fps
    
    @cap_fps.setter
    def cap_fps(self, cap_fps:int) -> None:
        self._cap_fps = cap_fps

    
    @property
    def start_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self._start_line
    
    @start_line.setter
    def start_line(self, start_line:Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        self._start_line = start_line

    
    @property
    def finish_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        return self._finish_line
    
    @finish_line.setter
    def finish_line(self, finish_line:Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        self._finish_line = finish_line



    @property
    def perspective_points(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        return self._perspective_points
    
    @perspective_points.setter
    def perspective_points(self, perspective_points: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> None:
        self._perspective_points = perspective_points


    @property
    def perspective_width(self) -> Optional[int]:
        return self._perspective_width
    
    @perspective_width.setter
    def perspective_width(self, perspective_width: int) -> None:
        self._perspective_width = perspective_width

    @property
    def perspective_height(self) -> Optional[int]:
        return self._perspective_height
    
    @perspective_height.setter
    def perspective_height(self, perspective_height: int) -> None:
        self._perspective_height = perspective_height

    @property
    def perspective_frame(self) -> Optional[cv.typing.MatLike]:
        return self._perspective_frame

    @property
    def perspective_matrix(self) -> Optional[cv.typing.MatLike]:
        return self._perspective_matrix



    def open(self) -> bool:

        if self._cap is not None and self._cap.isOpened():
            return True
        
        if self._cap_index is not None:
            try: 
                self._cap = cv.VideoCapture(self._cap_index, self._cap_api)
            except Exception as e:
                print(f"VideoCapture Cap Index: {self._cap} {e}")

        elif self._cap_source is not None:
            try:
                self._cap = cv.VideoCapture(self._cap_source, self._cap_api)
            except Exception as e:
                print(f"VideoCapture Cap Source: {self._cap_source} {e}")

        else:
            print(f"Index: {self._cap_index}, Source: {self._cap_source} Camera: Either cap_index or cap_source must be provided.")
            return False

        if self._cap is None:
            print(f"Camera {self._cap_index if self._cap_index is not None else self._cap_source}: Failed to open capture.")
            return False


        
        #if self._cap_width:
        #    self._cap.set(cv.CAP_PROP_FRAME_WIDTH, self._cap_width)
        #if self._cap_height:
        #    self._cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._cap_height)
        #if self._cap_fps:
        #    self._cap.set(cv.CAP_PROP_FPS, self._cap_fps)

        #for _ in range(10):
        #    self._cap.read()
        
        return True



    def read(self) -> bool:
        
        if self._cap is None or not self._cap.isOpened():
            if not self.open():
                blank = np.full((self._cap_height or 480, self._cap_width or 640, 3), 255, np.uint8)
                cv.putText(blank, f"Camera{self._cap_index if self._cap_index is not None else self._cap_source} is not open.", (240, 320), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
                self._frame = blank
                return False


        try:
            ret, frame = self._cap.read()
        except Exception as e:
            print(f"Cap Read: {e}")
        
        if not ret or frame is None:
            blank = np.full((self._cap_height or 480, self._cap_width or 640, 3), 255, np.uint8)
            cv.putText(blank, f"Camera{self._cap_index if self._cap_index is not None else self._cap_source}. No Frame", (240, 320), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2, cv.LINE_AA)
            self._frame = blank
            return False
        

        self._frame = frame

        return True
      
        
     
       

        
        
    def get_frame(self) -> Optional[cv.typing.MatLike]:
        self.read()
        return self._frame



    def release(self) -> None:

        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                print(f"Cap Release: {e}")
    
            self._cap = None


    def set_perspective_frame(self) -> bool:

        if self._frame is None:
            return False
        
        if not self._perspective_points:
            return False
        
        ((tl_x, tl_y), (tr_x, tr_y), (bl_x, bl_y), (br_x, br_y)) = self._perspective_points
        


        if self._perspective_width is None or self._perspective_height is None:
            x_max = max(abs(tl_x - tr_x), abs(bl_x - br_x))
            y_max = max(abs(tl_y - bl_y), abs(tr_y - br_y))

            self._perspective_width = x_max
            self._perspective_height = y_max



        if self._perspective_matrix is None:
        

            pts_src = np.array([[tl_x, tl_y], 
                                [tr_x, tr_y], 
                                [br_x, br_y],
                                [bl_x, bl_y]], dtype=np.float32)

            pts_dst = np.array([[0, 0], 
                                [self._perspective_width, 0], 
                                [self._perspective_width, self._perspective_height], 
                                [0, self._perspective_height]], dtype=np.float32)


            self._perspective_matrix = cv.getPerspectiveTransform(pts_src, pts_dst)


        warped = cv.warpPerspective(self._frame, self._perspective_matrix, (self._perspective_width, self._perspective_height))

        if self._start_line is not None and self._finish_line is not None and warped is not None:
            cv.line(warped, self._start_line[0], self._start_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)
            cv.line(warped, self._finish_line[0], self._finish_line[1], COLORS_BGR["Red"], 2, cv.LINE_AA)

        self._perspective_frame = warped

        return True





    def get_perspective_frame(self) -> Optional[cv.typing.MatLike]:

        if self._frame is None:
            print(f"Camera{self._cap_index if self._cap_index is not None else self._cap_source}: No Frame")
            return None
        
        if not (self._perspective_top_left and self._perspective_top_right and
                self._perspective_bottom_left and self._perspective_bottom_right):
            print(f"Camera{self._cap_index if self._cap_index is not None else self._cap_source}: no perspective values")
            return None
        
        self.set_perspective_frame()

        return self._perspective_frame