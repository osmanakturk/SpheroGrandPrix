async function startLap() {
    const res = await fetch("/lap/start", { method: "POST" });
    //const data = await res.json();

}


async function stopLap() {
    const res = await fetch("/lap/stop", { method: "POST" });
    //const data = await res.json();
    let total_games = document.getElementById("total_games");
    let total_spheros = document.getElementById("total_spheros");
    let total_red = document.getElementById("total_red");
    let total_yellow = document.getElementById("total_yellow");
    let total_blue = document.getElementById("total_blue");
    let total_green = document.getElementById("total_green");
    let best_score = document.getElementById("best_score");
    let mean_score = document.getElementById("mean_score");
    let last_score = document.getElementById("last_score");
    
}

async function reset(color) {
    const res = await fetch(`/reset/${color}`, { method: "POST" });
    //const data = await res.json();
}


async function changeUsername(color) {
    let element = document.getElementById(`username_${color}`);
    let username = (element?.value ?? "").trim();
    const res = await fetch(`/username_change/${encodeURIComponent(color)}/${encodeURIComponent(username)}`, { method:"POST" });
    //const data = await res.json();
}



async function lapStatus() {
    let red_username = document.getElementById("red_username");
    let red_start_time = document.getElementById("red_start_time");
    let red_finish_time = document.getElementById("red_finish_time");
    let red_lap_time = document.getElementById("red_lap_time");

    let yellow_username = document.getElementById("yellow_username");
    let yellow_start_time = document.getElementById("yellow_start_time");
    let yellow_finish_time = document.getElementById("yellow_finish_time");
    let yellow_lap_time = document.getElementById("yellow_lap_time");

    let blue_username = document.getElementById("blue_username");
    let blue_start_time = document.getElementById("blue_start_time");
    let blue_finish_time = document.getElementById("blue_finish_time");
    let blue_lap_time = document.getElementById("blue_lap_time");

    let green_username = document.getElementById("green_username");
    let green_start_time = document.getElementById("green_start_time");
    let green_finish_time = document.getElementById("green_finish_time");
    let green_lap_time = document.getElementById("green_lap_time");

    try {
        const res = await fetch("/lap/state", { method: "POST" });
        const data = await res.json();  


        //red_username.textContent = data.red.username
        red_start_time.textContent = data.red.start_time
        red_finish_time.textContent = data.red.finish_time
        red_lap_time.textContent = data.red.total_lap_time + " sec"

        //yellow_username.textContent = data.yellow.username
        yellow_start_time.textContent = data.yellow.start_time
        yellow_finish_time.textContent = data.yellow.finish_time
        yellow_lap_time.textContent = data.yellow.total_lap_time + " sec"


        //blue_username.textContent = data.blue.username
        blue_start_time.textContent = data.blue.start_time
        blue_finish_time.textContent = data.blue.finish_time
        blue_lap_time.textContent = data.blue.total_lap_time + " sec"

        //green_username.textContent = data.green.username
        green_start_time.textContent = data.green.start_time
        green_finish_time.textContent = data.green.finish_time
        green_lap_time.textContent = data.green.total_lap_time + " sec"

    } catch (error) {

        console.error(error)
    }
    
}


setInterval(lapStatus, 500)