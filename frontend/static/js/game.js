let chart_sphero = null

async function startLap() {

    const res = await fetch("/lap/start", { method: "POST" });
    if (!res.ok) {
        console.error(res.status)
    }
    const data = await res.json();
    console.log(`Start Lap: ${data.ok}`)

}


async function drawPlot(totals) {
    try {

        let canvas_sphero = document.getElementById("canvas_sphero")

        let xValues = ["Red", "Yellow", "Blue", "Green"]
        let yValues = [totals.red || 0, totals.yellow || 0, totals.blue || 0, totals.green || 0]
        let barColors = ["rgb(252, 3, 3)", "rgb(232, 252, 3)", "rgb(3, 19, 252)", "rgb(30, 173, 25)"]

        const chart_exixt = Chart.getChart(canvas_sphero);

        if (chart_exixt) {

            chart_sphero.data.datasets[0].data = yValues;
            chart_sphero.update();

        } else {
            chart_sphero = new Chart(canvas_sphero, {
                type: "doughnut",
                //type: "pie",
                data: {
                    labels: xValues,
                    datasets: [{
                        backgroundColor: barColors,
                        data: yValues
                    }]
                },

                options: {
                    plugins: {
                        legend: { display: true },
                        title: { display: false }
                    }
                }
            });
        }

    } catch (e) {
        console.error(e)
    }



}




async function updateStats() {
    try {
        const res = await fetch("/stats", { cache: "no-store" })
        if (!res.ok) {
            console.error(res.status)
        }
        const totals = await res.json()

        let total_games = document.getElementById("total_games")
        let total_spheros = document.getElementById("total_spheros")
        let total_red = document.getElementById("total_red");
        let total_yellow = document.getElementById("total_yellow");
        let total_blue = document.getElementById("total_blue");
        let total_green = document.getElementById("total_green");
        let dashboard_list = document.getElementById("dashboard_list")


        total_games.textContent = totals.games ?? 0
        total_spheros.textContent = totals.spheros ?? 0
        total_red.textContent = totals.red ?? 0
        total_yellow.textContent = totals.yellow ?? 0
        total_blue.textContent = totals.blue ?? 0
        total_green.textContent = totals.green ?? 0

        totals.dashboard.forEach(element => {
            let btnNode = document.createElement("button");
            btnNode.type = "button"
            btnNode.classList.add("btn", "btn-warning", "text-start", "list-group-item", "my-1", "border", "border-1", "border-black", "rounded-3")
            
            
            let textNode = document.createTextNode(element.result);
            btnNode.appendChild(textNode)

            dashboard_list.appendChild(btnNode)
            
        });



        await drawPlot(totals)

    } catch (e) {
        console.error(e)
    }


}



updateStats()


async function stopLap() {
    const res = await fetch("/lap/stop", { method: "POST" });
    if (!res.ok) {
        console.error(res.status)
    }
    const data = await res.json();
    

    

    let best_score = document.getElementById("best_score")
    best_score.textContent = data.score.best_score

    let mean_score = document.getElementById("mean_score")
    mean_score.textContent = data.score.mean_score

    let last_score = document.getElementById("last_score")
    last_score.textContent = data.score.last_score

    let lap_results = document.getElementById("lap_results")
    

    data.sphero.forEach((element, index) => {
        lap_results.innerHTML += `<div class="col-2 bg-danger rounded-5 border border-5 border-black p-2 d-flex flex-column justify-content-center align-items-center">
                                    <span class="my-1 fs-1 fw-bolder text-info">Score: ${index+1}</span>
                                    <span class="my-1 fs-6 fw-bolder text-info">Username: ${element.username}</span>
                                    <span class="my-1 fs-6 fw-bolder text-info">Start Time: ${element.start_time}</span>
                                    <span class="my-1 fs-6 fw-bolder text-info">Finish Time: ${element.finish_time}</span>
                                    <span class="my-1 fs-6 fw-bolder text-info">Lap Time: ${element.lap_time}</span>
                                </div>`
    });



    await updateStats()
}


async function reset(color) {

    const res = await fetch(`/reset/${encodeURIComponent(color)}`, { method: "POST" });
    if (!res.ok) {
        console.error(res.status)
    }
    const data = await res.json();
    console.log(`Reset ${color}: ${data.ok}`)
}


async function changeUsername(color) {
    let element = document.getElementById(`${color}_username`);
    
    
   
    let username = (element?.value ?? "").trim();
    if (username.length < 1) {
        username = color
    }
    const res = await fetch(`/username_change/${encodeURIComponent(color)}/${encodeURIComponent(username)}`, { method: "POST" });
     if (!res.ok) {
        console.error(res.status)
    }
    
    const data = await res.json();
    console.log(`Change Username: ${data.ok}, Color: ${color}, Username: ${username}`)

    element.value = ""
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
        const res = await fetch("/lap/state", { cache: "no-store" });
        if (!res.ok) {
            console.error(res.status)
        }
        const data = await res.json();


        //red_username.textContent = data.red.username
        red_start_time.textContent = data.red.start_time
        red_finish_time.textContent = data.red.finish_time
        red_lap_time.textContent = data.red.total_lap_time ? `${data.red.total_lap_time} sec` : '';

        //yellow_username.textContent = data.yellow.username
        yellow_start_time.textContent = data.yellow.start_time
        yellow_finish_time.textContent = data.yellow.finish_time
        yellow_lap_time.textContent = data.yellow.total_lap_time ? `${data.yellow.total_lap_time} sec` : '';


        //blue_username.textContent = data.blue.username
        blue_start_time.textContent = data.blue.start_time
        blue_finish_time.textContent = data.blue.finish_time
        blue_lap_time.textContent = data.blue.total_lap_time ? `${data.blue.total_lap_time} sec` : '';

        //green_username.textContent = data.green.username
        green_start_time.textContent = data.green.start_time
        green_finish_time.textContent = data.green.finish_time
        green_lap_time.textContent = data.green.total_lap_time ? `${data.green.total_lap_time} sec` : '';


    } catch (e) {

        console.error(e)
    }



}


setInterval(lapStatus, 500)