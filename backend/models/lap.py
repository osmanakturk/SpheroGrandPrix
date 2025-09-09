import uuid, sqlite3, time
import numpy as np
import cv2 as cv
from datetime import datetime
from backend.models.sphero_bolt import SpheroBolt
from backend.constants import PATH, HSV_RANGES_STRICT, HSV_RANGES_WIDE, COLORS_HSV, COLORS_BGR
from backend.utils import HsvColorsRange, SpheroColor
from typing import Optional, Tuple


class Lap:

    def __init__(self):
        

        self._id: Optional[str] = None
        self._is_started: bool = False
        self._is_stopped: bool = False
        self._debug_red: bool = False
        self._debug_yellow: bool = False
        self._debug_blue: bool = False
        self._debug_green: bool = False
        self._background_img: Optional[cv.typing.MatLike]  = None
        self._path_frame: Optional[cv.typing.MatLike]  = None
        self._finishline_frame: Optional[cv.typing.MatLike] = None
        self._sphero_bolt_red: Optional[SpheroBolt] = None
        self._sphero_bolt_green: Optional[SpheroBolt] = None
        self._sphero_bolt_yellow: Optional[SpheroBolt] = None
        self._sphero_bolt_blue: Optional[SpheroBolt] = None




    @property
    def path_frame(self) -> Optional[cv.typing.MatLike]:
        return self._path_frame
    

    @path_frame.setter
    def path_frame(self, path_frame: Optional[cv.typing.MatLike]) -> None:
        self._path_frame = path_frame.copy() if path_frame is not None else None

        for sphero in self._sphero_bolts():
            sphero.path_frame = self._path_frame

            if sphero.canvas is None:
                sphero.canvas = self.path_frame


    @property
    def finishline_frame(self) -> Optional[cv.typing.MatLike]:
        return self._finishline_frame
    
    @finishline_frame.setter
    def finishline_frame(self, finishline_frame: Optional[cv.typing.MatLike]) -> None:
        self._finishline_frame = finishline_frame.copy() if finishline_frame is not None else None

        for sphero in self._sphero_bolts():
            sphero.finishline_frame = self._finishline_frame


    @property
    def id(self) -> Optional[str]:
        if self._id is None:
            print("Lap is not started")
        return self._id
    
    @property 
    def is_started(self) -> bool:
        return self._is_started
    
    @is_started.setter
    def is_started(self, is_started: bool) -> None:
        self._is_started = is_started

        for sphero in self._sphero_bolts():
            sphero.is_lap_started = is_started


    @property 
    def is_stopped(self) -> bool:
        return self._is_stopped
    
    @is_stopped.setter
    def is_stopped(self, is_stopped: bool) -> None:
        self._is_stopped = is_stopped

        for sphero in self._sphero_bolts():
            sphero.is_lap_stopped = is_stopped


    @property
    def background_img(self) -> Optional[cv.typing.MatLike]:
        return self._background_img
    
    @background_img.setter
    def background_img(self, background_img: Optional[cv.typing.MatLike]) -> None:
        self._background_img = background_img.copy() if background_img is not None else None

        for sphero in self._sphero_bolts():
            sphero.background = self._background_img


    @property
    def debug_red(self) -> bool:
        return self._debug_red
    
    @debug_red.setter
    def debug_red(self, debug_red: bool) -> None:
        self._debug_red = debug_red
        self.sphero_bolt_red.debug = debug_red


    @property
    def debug_yellow(self) -> bool:
        return self._debug_yellow
    
    @debug_yellow.setter
    def debug_yellow(self, debug_yellow: bool) -> None:
        self._debug_yellow = debug_yellow
        self.sphero_bolt_yellow.debug = debug_yellow



    @property
    def debug_blue(self) -> bool:
        return self._debug_blue
    
    @debug_blue.setter
    def debug_blue(self, debug_blue: bool) -> None:
        self._debug_blue = debug_blue
        self.sphero_bolt_blue.debug = debug_blue


    @property
    def debug_green(self) -> bool:
        return self._debug_green
    
    @debug_green.setter
    def debug_green(self, debug_green: bool) -> None:
        self._debug_green = debug_green
        self.sphero_bolt_green.debug = debug_green


    @property
    def sphero_bolt_green(self) -> SpheroBolt:
        return self._sphero_bolt_green
    
    @property
    def sphero_bolt_yellow(self) -> SpheroBolt:
        return self._sphero_bolt_yellow
    
    @property
    def sphero_bolt_red(self) -> SpheroBolt:
        return self._sphero_bolt_red
    
    @property
    def sphero_bolt_blue(self) -> SpheroBolt:
        return self._sphero_bolt_blue
    


  
    def _start_impl(self,
              path_frame: Optional[cv.typing.MatLike]  = None,
              finishline_frame: Optional[cv.typing.MatLike] = None, 
              background_img: Optional[cv.typing.MatLike] = None, 
              debug_red: Optional[bool] = False, 
              debug_yellow: Optional[bool] = False, 
              debug_blue: Optional[bool] = False, 
              debug_green: Optional[bool] = False) -> bool:

        if self._is_started and not self._is_stopped:
            return False
            

        
        self._id = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}_{uuid.uuid4().hex}"
        self._is_started = True
        self._is_stopped = False

        self._path_frame = path_frame.copy() if path_frame is not None else None
        self._finishline_frame = finishline_frame.copy() if finishline_frame is not None else None
        self._background_img = background_img.copy() if background_img is not None else None
        self._debug_red = debug_red
        self._debug_yellow = debug_yellow
        self._debug_blue = debug_blue
        self._debug_green = debug_green



        self._sphero_bolt_red = SpheroBolt(
            lap_id=self._id, 
            color=SpheroColor.RED, 
            path_frame=self._path_frame.copy() if self._path_frame is not None else None, 
            finishline_frame=self._finishline_frame.copy() if self._finishline_frame is not None else None, 
            background=self._background_img.copy() if self._background_img is not None else None, 
            is_lap_started=self._is_started, 
            is_lap_stopped=self._is_stopped, 
            debug=self._debug_red
            )



        self._sphero_bolt_yellow = SpheroBolt(
            lap_id=self._id, 
            color=SpheroColor.YELLOW, 
            path_frame=self._path_frame.copy() if self._path_frame is not None else None, 
            finishline_frame=self._finishline_frame.copy() if self._finishline_frame is not None else None, 
            background=self._background_img.copy() if self._background_img is not None else None, 
            is_lap_started=self._is_started, 
            is_lap_stopped=self._is_stopped, 
            debug=self._debug_yellow
            )



        self._sphero_bolt_blue = SpheroBolt(
            lap_id=self._id, 
            color=SpheroColor.BLUE, 
            path_frame=self._path_frame.copy() if self._path_frame is not None else None, 
            finishline_frame=self._finishline_frame.copy() if self._finishline_frame is not None else None, 
            background=self._background_img.copy() if self._background_img is not None else None, 
            is_lap_started=self._is_started, 
            is_lap_stopped=self._is_stopped, 
            debug=self._debug_blue
            )




        self._sphero_bolt_green = SpheroBolt(
            lap_id=self._id, 
            color=SpheroColor.GREEN, 
            path_frame=self._path_frame.copy() if self._path_frame is not None else None, 
            finishline_frame=self._finishline_frame.copy() if self._finishline_frame is not None else None,  
            background=self._background_img.copy() if self._background_img is not None else None, 
            is_lap_started=self._is_started, 
            is_lap_stopped=self._is_stopped, 
            debug=self._debug_green
            )
        
        return True

        
    @classmethod
    def start(cls, 
              path_frame: Optional[cv.typing.MatLike]  = None,
              finishline_frame: Optional[cv.typing.MatLike] = None, 
              background_img: Optional[cv.typing.MatLike] = None, 
              debug_red: Optional[bool] = False, 
              debug_yellow: Optional[bool] = False, 
              debug_blue: Optional[bool] = False, 
              debug_green: Optional[bool] = False) -> "Lap":
        
        lap = cls()
        
        lap._start_impl(path_frame=path_frame, 
                        finishline_frame=finishline_frame, 
                        background_img= background_img, 
                        debug_red=debug_red, 
                        debug_yellow=debug_yellow,
                        debug_blue=debug_blue, 
                        debug_green=debug_green)
        print(f"Lap started with id: {lap._id}")
        return lap





    def _sphero_bolts(self) -> Tuple[SpheroBolt, SpheroBolt, SpheroBolt, SpheroBolt]:
        
        return (self._sphero_bolt_red, 
                self._sphero_bolt_yellow, 
                self._sphero_bolt_green, 
                self._sphero_bolt_blue)


    def get_processed_path_frame(
            self, 
            hsv_ranges: HsvColorsRange = HsvColorsRange.NORMAL, 
            min_radius: Optional[int] = None, 
            max_radius: Optional[int] = None, 
            bilateral_diameter: int = 9,
            bilateral_sigma_color: int = 75,
            bilateral_sigma_space: int = 75,
            median_kernel_size: int = 9,
            clahe_clip_limit : float = 4.0,
            clahe_tile_grid_size : int = 9,
            morph_kernel_size: int = 5,
            morph_iterator: int = 1,
            contours_chain_approx_simple: bool = True
            ) -> Optional[cv.typing.MatLike]:

        if not self._is_started:
            print("Lap not started yet")
            return None

        yellow = self._sphero_bolt_yellow.get_processed_path_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )


        red = self._sphero_bolt_red.get_processed_path_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )


        green = self.sphero_bolt_green.get_processed_path_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )
        
        blue = self.sphero_bolt_blue.get_processed_path_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )
        

        if yellow is not None or red is not None:
            if red is not None and yellow is not None:
                yellow_red = cv.bitwise_or(red, yellow)
            elif red is not None:
                yellow_red = red
            elif yellow is not None:
                yellow_red = yellow
        else:
            yellow_red = None


        if blue is not None or green is not None:
            if blue is not None and green is not None:
                blue_green = cv.bitwise_or(blue, green)
            elif blue is not None:
                blue_green = blue
            elif green is not None:
                blue_green = green
        else:
            blue_green = None
        

        if yellow_red is not None or blue_green is not None:
            if yellow_red is not None and blue_green is not None:
                frame = cv.bitwise_or(yellow_red, blue_green)
            elif yellow_red is not None:
                frame = yellow_red
            elif blue_green is not None:
                frame = blue_green
        else:
            frame = None


        return frame





    def get_processed_finishline_frame(
            self, 
            hsv_ranges: HsvColorsRange = HsvColorsRange.NORMAL, 
            min_radius: Optional[int] = None, 
            max_radius: Optional[int] = None,
            start_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
            finish_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None, 
            bilateral_diameter: int = 9,
            bilateral_sigma_color: int = 75,
            bilateral_sigma_space: int = 75,
            median_kernel_size: int = 9,
            clahe_clip_limit : float = 4.0,
            clahe_tile_grid_size : int = 9,
            morph_kernel_size: int = 5,
            morph_iterator: int = 1,
            contours_chain_approx_simple: bool = True
            ) -> Optional[cv.typing.MatLike]:

        
        if not self._is_started:
            print("Lap not started yet")
            return None

        yellow = self._sphero_bolt_yellow.get_processed_finishline_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius,
            start_line=start_line, 
            finish_line=finish_line, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )
        

        red = self._sphero_bolt_red.get_processed_finishline_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius,
            start_line=start_line, 
            finish_line=finish_line, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )


        green = self.sphero_bolt_green.get_processed_finishline_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius,
            start_line=start_line, 
            finish_line=finish_line, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )
        

        blue = self.sphero_bolt_blue.get_processed_finishline_frame(
            hsv_ranges=hsv_ranges, 
            min_radius=min_radius, 
            max_radius=max_radius,
            start_line=start_line, 
            finish_line=finish_line, 
            bilateral_diameter=bilateral_diameter,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            median_kernel_size=median_kernel_size,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid_size=clahe_tile_grid_size,
            morph_kernel_size=morph_kernel_size,
            morph_iterator=morph_iterator,
            contours_chain_approx_simple=contours_chain_approx_simple
            )
        

        if yellow is not None or red is not None:
            if red is not None and yellow is not None:
                yellow_red = cv.bitwise_or(red, yellow)
            elif red is not None:
                yellow_red = red
            elif yellow is not None:
                yellow_red = yellow
        else:
            yellow_red = None


        if blue is not None or green is not None:
            if blue is not None and green is not None:
                blue_green = cv.bitwise_or(blue, green)
            elif blue is not None:
                blue_green = blue
            elif green is not None:
                blue_green = green
        else:
            blue_green = None
        

        if yellow_red is not None or blue_green is not None:
            if yellow_red is not None and blue_green is not None:
                frame = cv.bitwise_or(yellow_red, blue_green)
            elif yellow_red is not None:
                frame = yellow_red
            elif blue_green is not None:
                frame = blue_green
        else:
            frame = None


        return frame


    def _save(self) -> bool:

        if not self._is_stopped:
            print("Lap could not be saved")
            return False

        try:

            with sqlite3.connect("database.sqlite") as db:

                cursor = db.cursor()

                cursor.execute("""CREATE TABLE IF NOT EXISTS sphero_bolt(
                               id TEXT PRIMARY KEY NOT NULL, 
                               lap_id TEXT NOT NULL, 
                               color TEXT NOT NULL, 
                               username TEXT NOT NULL, 
                               path_img_path TEXT, 
                               start_time TEXT, 
                               finish_time TEXT, 
                               total_lap_time REAL
                               )""")

                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sphero_bolt_lap_id ON sphero_bolt(lap_id)")


                sphero_bots = self._sphero_bolts()

                for sphero in sphero_bots:

                    if sphero is not None:

                        sphero_id = sphero.id 
                        lap_id = sphero.lap_id 
                        color = sphero.color.value 
                        username = sphero.username 
                        path_img_path = f"paths/{sphero.id}.png" if sphero.path_frame is not None and sphero.total_lap_time is not None else None
                        start_time = sphero.start_time.isoformat() if sphero.start_time is not None else None 
                        finish_time = sphero.finish_time.isoformat() if sphero.finish_time is not None else None 
                        total_lap_time = sphero.total_lap_time

                        cursor.execute("""INSERT INTO sphero_bolt 
                                       (id, lap_id, color, username, path_img_path, start_time, finish_time, total_lap_time) 
                                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", 
                                       (sphero_id, lap_id, color, username, path_img_path, start_time, finish_time, total_lap_time)
                                       )


                print("Lap saved")
            return True
        except Exception as e:
            print("Lap could not be saved")
            print(e)
            return False


    def stop(self) -> bool:
        
        if not self._is_started:
            print("Lap not started yet")
            return False
        
        print(f"Lap stoppen with id: {self._id}")
        
        self._is_stopped = True
        self._save()
        self._is_started = False
        self._sphero_bolt_red = None
        self._sphero_bolt_yellow = None
        self._sphero_bolt_blue = None
        self._sphero_bolt_green = None
        self._background_img = None
        self._id = None
        return True