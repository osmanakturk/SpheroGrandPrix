from flask import Flask, render_template, url_for, send_from_directory, request, redirect, Response, jsonify
from typing import Generator, Optional, Tuple, Literal
from backend.models.lap import Lap
import threading, atexit, time, json, sqlite3, sys, os
import numpy as np
from backend.enums import CaptureApi, HsvColorsRange
from backend.configs import CameraConfig, DetectorConfig
from backend.services.camera_tracker import (start_tracker,  
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



LOCK = threading.Lock()





is_tracker_started = False


def ensure_tracker_started() -> bool:
    global is_tracker_started

   
    if not is_tracker_started:
        is_tracker_started = start_tracker(
            
            finishline_cap_config=CameraConfig(
                cap_api=CaptureApi.Windows, 
                cap_index=2, 
                perspective_points=((93, 0), (510, 0), (99, 480), (502, 480)),
                start_line= ((0, 285), (192, 285)), 
                finish_line=((234, 285), (417, 285))
                ), 

            status_cap_config=CameraConfig(
                cap_api=CaptureApi.Windows, 
                cap_index=1
                )
            )
            
    return is_tracker_started


def ensure_get_tracker():
    back_points=((407, 51), (378, 331), (536, 242))
    middle_points=((407, 51), (323, 90), (314, 278), (443, 384), (452, 140))
    front_points= ((407, 51), (175, 424), (378, 331))

    finishline_detector_config = DetectorConfig(
        hsv_ranges=HsvColorsRange.MANUAL, 
        min_radius=15, 
        max_radius=30, 
        bilateral_diameter=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        median_kernel_size=9,
        clahe_clip_limit=4,
        clahe_tile_grid_size=9,
        morph_kernel_size=5,
        morph_iterator=1,
        contours_chain_approx_simple=True
    )


    return get_tracker(back_points=back_points, 
                       middle_points=middle_points, 
                       front_points=front_points, 
                       finishline_detector_config=finishline_detector_config, 
                       debug=False)



def get_lap() -> Lap:
    ensure_tracker_started()

    _, _, lap = ensure_get_tracker()

    return lap


def stream_status() -> Generator[bytes, None, None]:
    ensure_tracker_started()

    boundary = b"--frame"

    while True:
        status_jpg, _, _ = ensure_get_tracker()

        if status_jpg is None:
            continue
        
        yield(boundary + b"\r\n" 
              + b"Content-Type: image/jpeg"
              + b"\r\nContent-Length: " + str(len(status_jpg)).encode() 
              +b"\r\n\r\n" + status_jpg + b"\r\n") 


def stream_finishline() -> Generator[bytes, None, None]:
    ensure_tracker_started()

    boundary = b"--frame"

    while True:
        _, finishline_jpg, _ = ensure_get_tracker()

        if finishline_jpg is None:
            continue
        
        yield(boundary + b"\r\n" 
              + b"Content-Type: image/jpeg"
              + b"\r\nContent-Length: " + str(len(finishline_jpg)).encode() 
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
    return render_template("game.html", lap=lap)



@app.route("/video_feed/status")
def video_feed_status():
    return Response(stream_status(),  mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/video_feed/finishline")
def video_feed_finishline():
    return Response(stream_finishline(),  mimetype="multipart/x-mixed-replace; boundary=frame")
 

@app.route("/lap/start", methods=["POST"])
def start_lap_api():
    ok = lap_start()
    return jsonify({"ok": bool(ok)}), 200


@app.route("/lap/stop", methods=["POST"])
def stop_lap_api():

    lap_resuls = {"sphero": [], "score": {"best_score": 0, "last_score": 0, "mean_score": 0}}
    
    
    try:
        lap = get_lap()
    
   
        lap_id = lap.id
        red_username = lap.sphero_bolt_red.username.upper()
        red_start_time = lap.sphero_bolt_red.start_time
        red_finish_time = lap.sphero_bolt_red.finish_time
        red_lap_time = lap.sphero_bolt_red.total_lap_time
        

        yellow_username = lap.sphero_bolt_yellow.username.upper()
        yellow_start_time = lap.sphero_bolt_yellow.start_time
        yellow_finish_time = lap.sphero_bolt_yellow.finish_time
        yellow_lap_time = lap.sphero_bolt_yellow.total_lap_time
        

        blue_username = lap.sphero_bolt_blue.username.upper()
        blue_start_time = lap.sphero_bolt_blue.start_time
        blue_finish_time = lap.sphero_bolt_blue.finish_time
        blue_lap_time = lap.sphero_bolt_blue.total_lap_time
       

        green_username = lap.sphero_bolt_green.username.upper()
        green_start_time = lap.sphero_bolt_green.start_time
        green_finish_time = lap.sphero_bolt_green.finish_time
        green_lap_time = lap.sphero_bolt_green.total_lap_time
        


        if red_lap_time is not None:
            lap_resuls["sphero"].append({"username": red_username, "start_time": red_start_time.strftime("%H:%M:%S"), "finish_time": red_finish_time.strftime("%H:%M:%S"), "lap_time": red_lap_time,})
        if yellow_lap_time is not None:
            lap_resuls["sphero"].append({"username": yellow_username, "start_time": yellow_start_time.strftime("%H:%M:%S"), "finish_time": yellow_finish_time.strftime("%H:%M:%S"), "lap_time": yellow_lap_time})
        if blue_lap_time is not None:
            lap_resuls["sphero"].append({"username": blue_username, "start_time": blue_start_time.strftime("%H:%M:%S"), "finish_time": blue_finish_time.strftime("%H:%M:%S"), "lap_time": blue_lap_time})
        if green_lap_time is not None:
            lap_resuls["sphero"].append({"username": green_username, "start_time": green_start_time.strftime("%H:%M:%S"), "finish_time": green_finish_time.strftime("%H:%M:%S"), "lap_time": green_lap_time})

  

        

        if len(lap_resuls["sphero"]) > 0:

            lap_resuls["sphero"].sort(key= lambda x: x["lap_time"])

            lap_best = lap_resuls["sphero"][0]["lap_time"]
            lap_last = lap_resuls["sphero"][-1]["lap_time"]

            total_lap_time = 0.0

            for result in lap_resuls["sphero"]:
                total_lap_time += result["lap_time"]

            lap_mean = total_lap_time / len(lap_resuls["sphero"])

            lap_resuls["score"] = {"best_score": lap_best, "last_score": lap_last, "mean_score": lap_mean}
         
        
      
    except Exception as e:
        
        print(e)
    finally:
        ok = lap_stop()
      


    
    return jsonify(lap_resuls), 200




@app.route("/lap/state")
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
    
    resp = jsonify(data)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp, 200
 


@app.route('/stats')
def stats():

    totals = {
        "games": 0, 
        "spheros": 0, 
        "red": 0, 
        "yellow": 0, 
        "blue": 0, 
        "green": 0,
        "dashboard": []
    }


    if os.path.exists("database.sqlite"):
        with sqlite3.connect("database.sqlite") as db:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM sphero_bolt")

            data = cursor.fetchall()

            if len(data) > 0:
                data = np.array(data)

                totals["games"] = len(data) // 4

                spheros = data[data[:, 7] != None]
                totals["spheros"] = len(spheros)
                totals["red"] = len(spheros[spheros[:, 2] == "Red"])
                totals["yellow"] = len(spheros[spheros[:, 2] == "Yellow"])
                totals["blue"] = len(spheros[spheros[:, 2] == "Blue"])
                totals["green"] = len(spheros[spheros[:, 2] == "Green"])

                dashboard = spheros[spheros[:, 7].astype(float).argsort()]

                for sphero in dashboard:
                    username = str(sphero[3]).upper()
                    lap_time = float(sphero[7])
                    
                    lap_id = str(sphero[1])
                    arr = lap_id.split("_")
                    date = f"{arr[0]}/{arr[1]}/{arr[2]}"
                    time = f"{arr[3]}:{arr[4]}"
                    result = f"{username} {lap_time} sec ({date} {time})"
                    totals["dashboard"].append({"result": result})
                

    resp = jsonify(totals)
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp, 200




@app.route("/reset/<string:color>", methods=["POST"])
def reset_api(color):

    if color == "red":
        ok = reset_red()
    elif color == "yellow":
        ok = reset_yellow()
    elif color == "blue":
        ok = reset_blue()
    elif color == "green":
        ok = reset_green()
    else:
        return jsonify({"ok": False}), 400

    return jsonify({"ok": bool(ok)}), 200




@app.route("/username_change/<string:color>/<string:username>", methods=["POST"])
def username_change_api(color, username):
    if color == "red":
        ok = change_username_red(username)
    elif color == "yellow":
        ok = change_username_yellow(username)
    elif color == "blue":
        ok = change_username_blue(username)
    elif color == "green":
        ok = change_username_green(username)
    else:
        return jsonify({"ok": False}), 400

    return jsonify({"ok": bool(ok)}), 200




@app.route("/paths/<path:filename>")
def img_path(filename):
    return send_from_directory("paths", filename)


@app.route("/release_caps")
def release_caps():
    release_all()
    print("Cameras released")
    return redirect("/")


@app.route("/delete/database")
def delete_database():
    if os.path.exists("database.sqlite"):
        os.remove("database.sqlite")
        print("Database deleted")
    else:
        print("Database not exists")
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)