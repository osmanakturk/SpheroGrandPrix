import os
import numpy as np
import cv2 as cv
from datetime import datetime
from typing import Optional, Tuple
from backend.enums import SpheroColor
from backend.detectors.detector import Detector
from backend.configs import DetectorConfig, SpheroConfig





class SpheroBolt():

    def __init__(self, config:SpheroConfig):
    
        self._lap_id = config.lap_id
        self._color = config.color
        self._id = f"{config.lap_id}_{self._color.value}"
        self._username = self._color.value
        self._path_frame = config.path_frame.copy() if config.path_frame is not None else None
        self._finishline_frame = config.finishline_frame.copy() if config.finishline_frame is not None else None
        self._path_canvas = np.zeros_like(self._path_frame, np.uint8) if self._path_frame is not None else None
        self._finishline_canvas = np.zeros_like(self._finishline_frame, np.uint8) if self._finishline_frame is not None else None
        self._background = config.background.copy() if config.background is not None else None
        self._is_lap_started = config.is_lap_started
        self._is_lap_stopped = config.is_lap_stopped
        self._debug = config.debug
        self._is_started: bool = False
        self._is_finished: bool = False
        self._start_time: Optional[datetime] = None
        self._finish_time: Optional[datetime] = None
        self._total_lap_time: Optional[float] = None
        self._path_previous_center: Optional[Tuple[int, int]] = None
        self._path_center: Optional[Tuple[int, int]] = None
        self._path_radius: Optional[int] = None
        self._finishline_center: Optional[Tuple[int, int]] = None
        self._finishline_previous_center: Optional[Tuple[int, int]] = None
        self._finishline_radius: Optional[int] = None
        self._path_img = None
        
  


    @property
    def lap_id(self) -> str:
        return self._lap_id

    @lap_id.setter
    def lap_id(self, lap_id: str) -> None:
        self._lap_id = lap_id


    @property
    def id(self) -> str:
        return f"{self._lap_id}_{self._color.value}"
    
    @id.setter
    def id(self, lap_id: str) -> None:
        self._id = f"{lap_id}_{self._color.value}"

    @property
    def path_frame(self) -> Optional[cv.typing.MatLike]:
        return self._path_frame


    @path_frame.setter
    def path_frame(self, path_frame: Optional[cv.typing.MatLike]) -> None:
        self._path_frame = path_frame.copy() if path_frame is not None else None
        
        if self._path_canvas is None:
            self._path_canvas = np.zeros_like(path_frame, np.uint8) if path_frame is not None else None


    @property
    def finishline_frame(self) -> Optional[cv.typing.MatLike]:
        return self._finishline_frame


    @finishline_frame.setter
    def finishline_frame(self, finishline_frame: Optional[cv.typing.MatLike]) -> None:
        self._finishline_frame = finishline_frame.copy() if finishline_frame is not None else None

        if self._finishline_canvas is None:
            self._finishline_canvas = np.zeros_like(finishline_frame, np.uint8) if finishline_frame is not None else None

    @property
    def color(self) -> SpheroColor:
        return self._color

    @color.setter
    def color(self, color: SpheroColor) -> None:
        self._color = color

    @property
    def username(self) -> str:
        return self._username

    @username.setter
    def username(self, username: str) -> None:
        self._username = username


    @property
    def background(self) -> Optional[cv.typing.MatLike]:
        return self._background

    @background.setter
    def background(self, background: Optional[cv.typing.MatLike]) -> None:
        self._background = background.copy() if background is not None else None


    @property
    def is_started(self) -> bool:
        return self._is_started

    @is_started.setter
    def is_started(self, is_started: bool) -> None:
        self._is_started = is_started


    @property
    def is_finished(self) -> bool:
        return self._is_finished

    @is_finished.setter
    def is_finished(self, is_finished: bool) -> None:
        self._is_finished = is_finished


    @property
    def is_lap_started(self) -> bool:
        return self._is_lap_started

    @is_lap_started.setter
    def is_lap_started(self, is_lap_started: bool) -> None:
        self._is_lap_started = is_lap_started


    @property
    def is_lap_stopped(self) -> bool:
        return self._is_lap_stopped

    @is_lap_stopped.setter
    def is_lap_stopped(self, is_lap_stopped: bool) -> None:
        self._is_lap_stopped = is_lap_stopped



    @property
    def start_time(self) -> Optional[datetime]:
        return self._start_time

    @start_time.setter
    def start_time(self, start_time: Optional[datetime]) -> None:
        if self._is_started:
            self._start_time = start_time

    @property
    def finish_time(self) -> Optional[datetime]:
        return self._finish_time

    @finish_time.setter
    def finish_time(self, finish_time: Optional[datetime]) -> None:
        if self._is_finished:
            self._finish_time = finish_time

    @property
    def total_lap_time(self) -> Optional[float]:
        return self._total_lap_time

    @total_lap_time.setter
    def total_lap_time(self, total_lap_time: Optional[float]) -> None:
        if self._is_finished:
            self._total_lap_time = total_lap_time


    @property
    def path_previous_center(self) -> Optional[Tuple[int, int]]:
        return self._path_previous_center

    @path_previous_center.setter
    def path_previous_center(self, path_previous_center: Optional[Tuple[int, int]]) -> None:
        self._path_previous_center = path_previous_center

    @property
    def path_center(self) -> Optional[Tuple[int, int]]:
        return self._path_center

    @path_center.setter
    def path_center(self, path_center: Optional[Tuple[int, int]]) -> None:
        self._path_center = path_center

    @property
    def path_radius(self) -> Optional[int]:
        return self._path_radius

    @path_radius.setter
    def path_radius(self, path_radius: Optional[int]) -> None:
        self._path_radius = path_radius



    @property
    def finishline_center(self) -> Optional[Tuple[int, int]]: 
        return self._finishline_center

    @finishline_center.setter
    def finishline_center(self, finishline_center: Optional[Tuple[int, int]]) -> None:
        self._finishline_center = finishline_center


    @property
    def finishline_previous_center(self) -> Optional[Tuple[int, int]]: 
        return self._finishline_previous_center

    @finishline_previous_center.setter
    def finishline_previous_center(self, finishline_previous_center: Optional[Tuple[int, int]]) -> None:
        self._finishline_previous_center = finishline_previous_center



    @property
    def finishline_radius(self) -> Optional[int]:
        return self._finishline_radius

    @finishline_radius.setter
    def finishline_radius(self, finishline_radius: Optional[int]) -> None:
        self._finishline_radius = finishline_radius

    @property
    def path_img(self) -> Optional[cv.typing.MatLike]:
        return self._path_img


    @path_img.setter
    def path_img(self, path_img: Optional[cv.typing.MatLike]) -> None:
        self._path_img = path_img


    @property
    def path_canvas(self) -> Optional[cv.typing.MatLike]:
        return self._path_canvas

    @path_canvas.setter
    def path_canvas(self, frame: Optional[cv.typing.MatLike]) -> None:
        self._path_canvas = np.zeros_like(frame, np.uint8) if frame is not None else None


    @property
    def finishline_canvas(self) -> Optional[cv.typing.MatLike]:
        return self._finishline_canvas

    @finishline_canvas.setter
    def finishline_canvas(self, frame: Optional[cv.typing.MatLike]) -> None:
        self._finishline_canvas = np.zeros_like(frame, np.uint8) if frame is not None else None

    
    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, debug: bool) -> None:
        self._debug = debug
   

    def get_processed_path_frame(
            self, 
            config:DetectorConfig 
            ) -> Optional[cv.typing.MatLike]:
        
        config.sphero_bolt = self
        config.debug = self._debug

        return Detector.get_detected_path_frame(config=config)




    def get_processed_finishline_frame(
            self, 
            config:DetectorConfig, 
            start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
            finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]]
            ) -> Optional[cv.typing.MatLike]:
        
        config.sphero_bolt = self
        config.debug = self._debug

        return Detector.get_detected_finishline_frame(
            config=config, 
            start_line=start_line, 
            finish_line=finish_line
            )



    def save_path_img(self) -> bool:
        if self._path_canvas is None or self._total_lap_time is None:
            return False

        try:
            self._path_img = np.zeros((self._path_canvas.shape[0]+50, self._path_canvas.shape[1], 3), np.uint8)

            if self._background is not None:
                temp = cv.bitwise_or(self._path_canvas, self._background)
                self._path_img[50:, :] = temp[:, :]
            else:
                self._path_img[50:, :] = self._path_canvas[:, :]

            cv.rectangle(img=self._path_img, 
                         pt1=(0, 0), pt2=(self.path_img.shape[1]-1, 47), 
                         color=(0, 0, 255), 
                         thickness=1, 
                         lineType=cv.LINE_AA)

            cv.putText(img=self._path_img, 
                       text=f"User: {self._username}", 
                       org=(10, 20), fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, 
                       color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
            
            cv.putText(img=self._path_img, 
                       text=f"Lap Time: {self._total_lap_time} sec", 
                       org=(10, 40), fontFace=cv.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.5, 
                       color=(255, 255, 255), thickness=1, lineType=cv.LINE_AA)
            
            os.makedirs("paths", exist_ok=True)
            cv.imwrite(filename=f"paths/{self._id}.png", img=self._path_img)
            print(f"{self.id}.png saved")
            return True
        except Exception as e:
            print(f"{self.id}.png could not be saved")
            print(e)
            return False




    def reset(self) -> bool:
        
        #if self._is_finished or self._is_lap_stopped:
        #    return False
        
        #if not self._is_lap_started:
        #    return False

        try:
            self._is_started = False
            self._is_finished = False

            self._start_time = None
            self._finish_time = None
            self._total_lap_time = None
            
            self._path_canvas = np.zeros_like(self._path_canvas, dtype=np.uint8) if self._path_canvas is not None else None
            self._finishline_canvas = np.zeros_like(self._finishline_canvas, dtype=np.uint8) if self._finishline_canvas is not None else None

            self._path_center = None
            self._path_previous_center = None
            self._path_radius = None

            self._finishline_center = None
            self._finishline_previous_center = None
            self._finishline_radius = None
            
            if os.path.exists(f"paths/{self._id}.png"):
                os.remove(f"paths/{self._id}.png")
                print(f"{self._id}.png deleted")

            self._path_img = None

            print(f"{self._username} resetted")
            return True
        except Exception as e:
            print(f"{self._username} could not be resetted")
            print(e)
            return False