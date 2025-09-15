from flask import Flask, render_template, url_for, send_from_directory, request, redirect, Response
from flask_sqlalchemy import SQLAlchemy
from backend.models.sphero_bolt import SpheroBolt
from typing import Generator, Optional, Tuple, Literal
from backend.models.lap import Lap
import threading
from backend.trackers.camera_tracker import (start_tracker,  
                                             get_tracker, 
                                             lap_start, 
                                             lap_stop, 
                                             release_all, 
                                             reset_red, 
                                             reset_yellow, 
                                             reset_blue, 
                                             reset_green, 
                                             change_username_red, 
                                             change_username_yellow, 
                                             change_username_blue, 
                                             change_username_green)



app = Flask(__name__, 
            static_folder="frontend/static", 
            template_folder="frontend/templates")


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite'

LOCK = threading.Lock()




is_tracker_started = False


def ensure_tracker_started() -> bool:
    global is_tracker_started

    with LOCK:
        if not is_tracker_started:
            is_tracker_started = start_tracker()
    return is_tracker_started



def get_lap() -> Lap:
    ensure_tracker_started()

    _, _, lap = get_tracker()

    return lap

def stream_path() -> Generator[bytes, None, None]:
    ensure_tracker_started()

    boundary = b"--frame"

    while True:
        path_jpg, _, _ = get_tracker()

        if path_jpg is None:
            continue
        
        yield(boundary + b"\r\n" 
              + b"Content-Type: image/jpeg\r\n"
              + b"Content-Length: " + str(len(path_jpg)).encode() 
              +b"\r\n\r\n" + path_jpg + b"\r\\n") 


def stream_finishline() -> Generator[bytes, None, None]:
    ensure_tracker_started()

    boundary = b"--frame"

    while True:
        _, finishline_jpg, _ = get_tracker()

        if finishline_jpg is None:
            continue
        
        yield(boundary + b"\r\n" 
              + b"Content-Type: image/jpeg\r\n"
              + b"Content-Length: " + str(len(finishline_jpg)).encode() 
              +b"\r\n\r\n" + finishline_jpg + b"\r\\n") 




@app.route("/")
def home():
    lap = get_lap()
    return render_template("home.html", lap=lap)



@app.route("/video_feed/path")

def video_feed_path():
    return Response(stream_path(),  mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/video_feed/finishline")
def video_feed_finishline():
    return Response(stream_finishline(),  mimetype="multipart/x-mixed-replace; boundary=frame")
 



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)