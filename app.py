from flask import Flask, render_template, url_for, send_from_directory, request, redirect, Response, jsonify
from flask_sqlalchemy import SQLAlchemy
from backend.models.sphero_bolt import SpheroBolt
from typing import Generator, Optional, Tuple, Literal
from backend.models.lap import Lap
import threading, time, json, sqlite3, sys, os
import numpy as np
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
              +b"\r\n\r\n" + path_jpg + b"\r\n") 


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
              +b"\r\n\r\n" + finishline_jpg + b"\r\n") 











@app.route("/")
def home():
    return render_template("home.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")


@app.route("/game")
def game():
    lap = get_lap()

    total = {}
    total["games"] = 0
    total["spheros"] = 0
    total["red"] = 0
    total["yellow"] = 0
    total["blue"] = 0
    total["green"] = 0


    if os.path.exists("database.sqlite"):
        db = sqlite3.connect("database.sqlite")

        cursor = db.cursor()

        cursor.execute("SELECT * FROM sphero_bolt")

        data = cursor.fetchall()

        data = np.array(data)

        total["games"] = len(data) // 4

        spheros = data[data[:, 7] != None]

        total["spheros"] = len(spheros)
        total["red"] = len(spheros[spheros[:, 2] == "Red"])
        total["yellow"] = len(spheros[spheros[:, 2] == "Yellow"])
        total["blue"] = len(spheros[spheros[:, 2] == "Blue"])
        total["green"] = len(spheros[spheros[:, 2] == "Green"])
        





    return render_template("game.html", lap=lap, total=total)



@app.route("/video_feed/path")
def video_feed_path():
    return Response(stream_path(),  mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/video_feed/finishline")
def video_feed_finishline():
    return Response(stream_finishline(),  mimetype="multipart/x-mixed-replace; boundary=frame")
 

@app.route("/lap/start", methods=["POST"])
def start_lap_api():
    lap_start()
    return redirect("/game"), 200


@app.route("/lap/stop", methods=["POST"])
def stop_lap_api():
    lap_stop()
    
    return redirect("/game"), 200


@app.route("/lap/state", methods=["POST"])
def lap_state():
    lap = get_lap()
    
    data = {
        "red" : {
            "is_started": lap.sphero_bolt_red.is_started if lap.sphero_bolt_red is not None else False,
            "username": lap.sphero_bolt_red.username if lap.sphero_bolt_red is not None else '', 
            "start_time": lap.sphero_bolt_red.start_time.strftime('%H:%M:%S') if lap.sphero_bolt_red is not None and lap.sphero_bolt_red.start_time is not None else '', 
            "finish_time": lap.sphero_bolt_red.finish_time.strftime('%H:%M:%S') if lap.sphero_bolt_red is not None and lap.sphero_bolt_red.finish_time is not None else '', 
            "total_lap_time": lap.sphero_bolt_red.total_lap_time if lap.sphero_bolt_red is not None and lap.sphero_bolt_red.total_lap_time is not None else ''
            }, 
        
        "yellow": {
            "is_started": lap.sphero_bolt_yellow.is_started if lap.sphero_bolt_yellow is not None else False,
            "username": lap.sphero_bolt_yellow.username if lap.sphero_bolt_yellow is not None else '', 
            "start_time": lap.sphero_bolt_yellow.start_time.strftime('%H:%M:%S') if lap.sphero_bolt_yellow is not None and lap.sphero_bolt_yellow.start_time is not None else '', 
            "finish_time": lap.sphero_bolt_yellow.finish_time.strftime('%H:%M:%S') if lap.sphero_bolt_yellow is not None and lap.sphero_bolt_yellow.finish_time is not None else '', 
            "total_lap_time": lap.sphero_bolt_yellow.total_lap_time if lap.sphero_bolt_yellow is not None and lap.sphero_bolt_yellow.total_lap_time is not None else ''
            }, 
        
        "blue": {
            "is_started": lap.sphero_bolt_blue.is_started if lap.sphero_bolt_blue is not None else False,
            "username": lap.sphero_bolt_blue.username if lap.sphero_bolt_blue is not None else '', 
            "start_time": lap.sphero_bolt_blue.start_time.strftime('%H:%M:%S') if lap.sphero_bolt_blue is not None and lap.sphero_bolt_blue.start_time is not None else '', 
            "finish_time": lap.sphero_bolt_blue.finish_time.strftime('%H:%M:%S') if lap.sphero_bolt_blue is not None and lap.sphero_bolt_blue.finish_time is not None else '', 
            "total_lap_time": lap.sphero_bolt_blue.total_lap_time if lap.sphero_bolt_blue is not None and lap.sphero_bolt_blue.total_lap_time is not None else ''
             },

        "green": {
            "is_started": lap.sphero_bolt_green.is_started if lap.sphero_bolt_green is not None else False,
            "username": lap.sphero_bolt_green.username if lap.sphero_bolt_green is not None else '', 
            "start_time": lap.sphero_bolt_green.start_time.strftime('%H:%M:%S') if lap.sphero_bolt_green is not None and lap.sphero_bolt_green.start_time is not None else '', 
            "finish_time": lap.sphero_bolt_green.finish_time.strftime('%H:%M:%S') if lap.sphero_bolt_green is not None and lap.sphero_bolt_green.finish_time is not None else '', 
            "total_lap_time": lap.sphero_bolt_green.total_lap_time if lap.sphero_bolt_green is not None and lap.sphero_bolt_green.total_lap_time is not None else ''
            }}
    

    return jsonify(data), 200
 



 

@app.route("/reset/<string:color>", methods=["POST"])
def reset_api(color):

    if color == "red":
        reset_red()
    elif color == "yellow":
        reset_yellow()
    elif color == "blue":
        reset_blue()
    elif color == "green":
        reset_green()


    return redirect("/game"), 200




@app.route("/username_change/<string:color>/<string:username>", methods=["POST"])

def username_change_api(color, username):


    if color == "red":
        change_username_red(username)
    elif color == "yellow":
        change_username_yellow(username)
    elif color == "blue":
        change_username_blue(username)
    elif color == "green":
        change_username_green(username)

    return redirect("/game"), 200







if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)