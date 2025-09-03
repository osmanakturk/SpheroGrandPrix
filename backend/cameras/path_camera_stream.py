import cv2 as cv
import numpy as np
from typing import Generator, Tuple, Optional
from datetime import datetime
import os


def get_path_camera_processed_stream(path_top_left: Optional[Tuple[int, int]] = None, 
                                    path_top_right: Optional[Tuple[int, int]] = None, 
                                    path_bottom_left: Optional[Tuple[int, int]] = None, 
                                    path_bottom_right: Optional[Tuple[int, int]] = None,
                                    path_write: bool = False,
                                    path_width: Optional[int] = None, 
                                    path_height: Optional[int] = None,
                                    cap_api: int = cv.CAP_DSHOW
                                    ) -> Generator[cv.typing.MatLike, None, None]:


    path_cap = cv.VideoCapture(2, cap_api)


    if not path_cap.isOpened():
        raise RuntimeError("Cannot open camera")

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


    path_writer = None
    if path_write:
        os.makedirs("camera_records", exist_ok=True)
        path_ts = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        path_writer = cv.VideoWriter(f"camera_records/path_camera_{path_ts}.mp4", cv.VideoWriter.fourcc(*"mp4v"),path_fps, (path_width, path_height))

        if not path_writer.isOpened():
            path_writer = None
            raise RuntimeError("Cannot open VideoWriter (path_camera.mp4)")
    

    try:
        while True:
            path_ret, path_frame = path_cap.read()

            if not path_ret:
                break

            if path_matrix is not None:
                path_frame =  cv.warpPerspective(path_frame, path_matrix, (path_width, path_height))

            if path_write and path_writer is not None:
                path_writer.write(path_frame)

            yield path_frame
    finally:
        path_cap.release()

        if path_writer is not None:
            path_writer.release()




if __name__=="__main__":

    try:
        for path_frame in get_path_camera_processed_stream():
            cv.imshow("Path Camera", path_frame)

            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv.destroyAllWindows()
