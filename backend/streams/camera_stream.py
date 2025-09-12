import cv2 as cv
import numpy as np
from typing import Generator, Tuple, Optional
from datetime import datetime
import os


def get_camera_processed_stream(
        path_top_left: Optional[Tuple[int, int]] = None, 
        path_top_right: Optional[Tuple[int, int]] = None, 
        path_bottom_left: Optional[Tuple[int, int]] = None, 
        path_bottom_right: Optional[Tuple[int, int]] = None,
        path_write: bool = False,
        path_width: Optional[int] = None, 
        path_height: Optional[int] = None,
        finishline_top_left: Optional[Tuple[int, int]] = None, 
        finishline_top_right: Optional[Tuple[int, int]] = None, 
        finishline_bottom_left: Optional[Tuple[int, int]] = None, 
        finishline_bottom_right: Optional[Tuple[int, int]] = None,
        finishline_write: bool = False,
        finishline_width: Optional[int] = None, 
        finishline_height: Optional[int] = None,
        cap_api: int = cv.CAP_DSHOW) -> Generator[Tuple[cv.typing.MatLike, cv.typing.MatLike], None, None]:


    finishline_cap = cv.VideoCapture(1, cap_api)
    path_cap = cv.VideoCapture(2, cap_api)

    if not finishline_cap.isOpened():
        raise RuntimeError("Cannot open finishline camera")
    
    if not path_cap.isOpened():
        raise RuntimeError("Cannot open path camera")
    

    finishline_fps = int(finishline_cap.get(cv.CAP_PROP_FPS))

    if finishline_fps <= 0:
        finishline_fps = 30

    
    path_fps = int(path_cap.get(cv.CAP_PROP_FPS))

    if path_fps <= 0:
        path_fps = 30

    if all([path_top_left, path_top_right, path_bottom_left, path_bottom_right]):
        path_tl_x, path_tl_y = path_top_left
        path_tr_x, path_tr_y = path_top_right
        path_bl_x, path_bl_y = path_bottom_left
        path_br_x, path_br_y = path_bottom_right

        if path_width is None or path_height is None:
            path_x_max = max(abs(path_tl_x - path_tr_x), abs(path_bl_x - path_br_x))
            path_y_max = max(abs(path_tl_y - path_bl_y), abs(path_tr_y - path_br_y))

            path_width = int(path_x_max)
            path_height = int(path_y_max)

        path_pts_src = np.array([[path_tl_x, path_tl_y], 
                            [path_tr_x, path_tr_y], 
                            [path_br_x, path_br_y],
                            [path_bl_x, path_bl_y]], dtype=np.float32)
            
        path_pts_dst = np.array([[0, 0], 
                            [path_width, 0], 
                            [path_width, path_height], 
                            [0, path_height]], dtype=np.float32)


        path_matrix = cv.getPerspectiveTransform(path_pts_src, path_pts_dst)
    else:
        path_matrix = None
        path_width = int(path_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        path_height = int(path_cap.get(cv.CAP_PROP_FRAME_HEIGHT))



    if all([finishline_top_left, finishline_top_right, finishline_bottom_left, finishline_bottom_right]):
        finishline_tl_x, finishline_tl_y = finishline_top_left
        finishline_tr_x, finishline_tr_y = finishline_top_right
        finishline_bl_x, finishline_bl_y = finishline_bottom_left
        finishline_br_x, finishline_br_y = finishline_bottom_right

        if finishline_width is None or finishline_height is None:
            finishline_x_max = max(abs(finishline_tl_x - finishline_tr_x), abs(finishline_bl_x - finishline_br_x))
            finishline_y_max = max(abs(finishline_tl_y - finishline_bl_y), abs(finishline_tr_y - finishline_br_y))

            finishline_width = int(finishline_x_max)
            finishline_height = int(finishline_y_max)

        finishline_pts_src = np.array([[finishline_tl_x, finishline_tl_y], 
                            [finishline_tr_x, finishline_tr_y], 
                            [finishline_br_x, finishline_br_y],
                            [finishline_bl_x, finishline_bl_y]], dtype=np.float32)
            
        finishline_pts_dst = np.array([[0, 0], 
                            [finishline_width, 0], 
                            [finishline_width, finishline_height], 
                            [0, finishline_height]], dtype=np.float32)


        finishline_matrix = cv.getPerspectiveTransform(finishline_pts_src, finishline_pts_dst)
    else:
        finishline_matrix = None
        finishline_width = int(finishline_cap.get(cv.CAP_PROP_FRAME_WIDTH))
        finishline_height = int(finishline_cap.get(cv.CAP_PROP_FRAME_HEIGHT))



    ts = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

    finishline_writer = None
    if finishline_write:
        os.makedirs("camera_records", exist_ok=True)
        finishline_writer = cv.VideoWriter(f"camera_records/finishline_camera_{ts}.mp4", cv.VideoWriter.fourcc(*"mp4v"),finishline_fps, (finishline_width, finishline_height))

        if not finishline_writer.isOpened():
            finishline_writer = None
            raise RuntimeError("Cannot open VideoWriter (finishline_camera.mp4)")



    path_writer = None
    if path_write:
        os.makedirs("camera_records", exist_ok=True)
        path_writer = cv.VideoWriter(f"camera_records/path_camera_{ts}.mp4", cv.VideoWriter.fourcc(*"mp4v"),path_fps, (path_width, path_height))

        if not path_writer.isOpened():
            path_writer = None
            raise RuntimeError("Cannot open VideoWriter (path_camera.mp4)")
        


    try:
        while True:
            finishline_ret, finishline_frame = finishline_cap.read()
            path_ret, path_frame = path_cap.read()


            if not finishline_ret:
                print("Finishline camera read failed")
                break


            if not path_ret:
                print("Path camera read failed")
                break


            if finishline_matrix is not None:
                finishline_frame =  cv.warpPerspective(finishline_frame, finishline_matrix, (finishline_width, finishline_height))

            if finishline_write and finishline_writer is not None:
                finishline_writer.write(finishline_frame)


            if path_matrix is not None:
                path_frame =  cv.warpPerspective(path_frame, path_matrix, (path_width, path_height))

            if path_write and path_writer is not None:
                path_writer.write(path_frame)

            yield (path_frame, finishline_frame)


    finally:
        finishline_cap.release()
        path_cap.release()


        if path_writer is not None:
            path_writer.release()

        if finishline_writer is not None:
            finishline_writer.release()




if __name__=="__main__":

    try:
        for (path_frame, finishline_frame)in get_camera_processed_stream():
            cv.imshow("Finishline Camera", finishline_frame)
            cv.imshow("Path Camera", path_frame)

            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv.destroyAllWindows()