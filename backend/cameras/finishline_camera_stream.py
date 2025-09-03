import cv2 as cv
import numpy as np
from typing import Generator, Tuple, Optional
from datetime import datetime
import os


def get_finishline_camera_processed_stream(
                                    finishline_top_left: Optional[Tuple[int, int]] = None, 
                                    finishline_top_right: Optional[Tuple[int, int]] = None, 
                                    finishline_bottom_left: Optional[Tuple[int, int]] = None, 
                                    finishline_bottom_right: Optional[Tuple[int, int]] = None,
                                    finishline_write: bool = False,
                                    finishline_width: Optional[int] = None, 
                                    finishline_height: Optional[int] = None,
                                    cap_api: Optional[int] = cv.CAP_DSHOW
                                    ) -> Generator[cv.typing.MatLike, None, None]:


    finishline_cap = cv.VideoCapture(1, cap_api)


    if not finishline_cap.isOpened():
        raise RuntimeError("Cannot open camera")

    finishline_fps = int(finishline_cap.get(cv.CAP_PROP_FPS))

    if finishline_fps <= 0:
        finishline_fps = 30


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


    finishline_writer = None
    if finishline_write:
        os.makedirs("camera_records", exist_ok=True)
        finishline_ts = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
        finishline_writer = cv.VideoWriter(f"camera_records/finishline_camera_{finishline_ts}.mp4", cv.VideoWriter.fourcc(*"mp4v"),finishline_fps, (finishline_width, finishline_height))

        if not finishline_writer.isOpened():
            finishline_writer = None
            raise RuntimeError("Cannot open VideoWriter (finishline_camera.mp4)")
    

    try:
        while True:
            finishline_ret, finishline_frame = finishline_cap.read()

            if not finishline_ret:
                break

            if finishline_matrix is not None:
                finishline_frame =  cv.warpPerspective(finishline_frame, finishline_matrix, (finishline_width, finishline_height))

            if finishline_write and finishline_writer is not None:
                finishline_writer.write(finishline_frame)

            yield finishline_frame
    finally:
        finishline_cap.release()

        if finishline_writer is not None:
            finishline_writer.release()




if __name__=="__main__":

    try:
        for finishline_frame in get_finishline_camera_processed_stream():
            cv.imshow("Finishline Camera", finishline_frame)

            if cv.waitKey(1) & 0xFF == 27:
                break
    finally:
        cv.destroyAllWindows()
