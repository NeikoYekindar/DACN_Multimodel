function loadView(view) {
    const container = document.getElementById("main-content")
    container.innerHTML = "";

    if (view === "home") {
        container.innerHTML = `
                <main class="main">
                    <h2 class="section-title">Overview</h2>
                <div class="overview">
                    <div class="map-box">
                        <div id="map" class="mapbox"></div>
                        <button class="map-label" onclick="loadView('map')">Full view</button>
                    </div>
                    <div class="status-box">
                        <h3 class="status-title">Operational Overview</h3>
                        <div class="status-grid">
                            <!-- Cột trái -->
                            <div class="status-item">
                                <div class="status-label">
                                    <img src="assets/camera.png" />
                                    <span>Camera Online</span>
                                </div>
                                <div class="status-count">5</div>
                            </div>
                            <!-- Cột phải -->
                            <div class="status-item">
                                <div class="status-label">
                                    <img src="assets/red.png" />
                                    <span>Red Zone</span>
                                </div>
                                <div class="status-count">3</div>
                            </div>
                            <!-- Cột trái dòng 2 -->
                            <div class="status-item">
                                <div class="status-label">
                                    <img src="assets/drone.png" />
                                    <span>Drone Online</span>
                                </div>
                              
                              <div class="status-count">0</div>
                            </div>
                            <!-- Cột phải dòng 2 -->
                            <div class="status-item">
                                <div class="status-label">
                                    <img src="assets/green.png" />
                                    <span>Green Zone</span>
                                </div>
                              <div class="status-count">2</div>
                            </div>
                        </div>
                        <button class="btn-detail">Detail</button> 
                    </div>
                </div>
                <div class="bottom-box">
                    <!-- Nội dung biểu đồ hoặc log -->
                </div>
                </main>
        `;
        initMiniMap();

    }
    if (view === "map") {
        container.innerHTML = `
        <main class="main">
            <div id = "map_2" class = "map-full"></div>
             <div class="popup" id="station_detail">
        <img src='./assets/back.png' id="btn_back">
        <h2 class>Station 1</h2>
        <p>Phường Linh Trung, Thủ Đức, Hồ Chí Minh, Việt Nam.</p>
        <p id="type">Flood risk</p>
        <div id="box_container">

            <div class="info-box">
                <div class="info-header">
                    <img src="./assets/camera.png" id="camera">
                    <p>Camera</p>
                </div>
                <div class="info-content">
                    <div class="info">
                        <p class="state">State: </p>
                        <p>Up</p>
                    </div>
                    <div class="info">
                        <p class="timestamp">Time stamp:</p>
                        <p>2025-04-15 9:15:53</p>
                    </div>
                </div>
            </div>

            <div class="info-box">
                <div class="info-header">
                    <img src="./assets/drone.png" id="drone">
                    <p>Drone</p>
                </div>
                <div class="info-content">
                    <div class="info">
                        <p class="state">State: </p>
                        <p>Up</p>
                    </div>
                    <div class="info">
                        <p class="battery">Battery: </p>
                        <p>87%</p>
                    </div>
                    <div class="info">
                        <p class="timestamp">Time stamp: </p>
                        <p>2025-04-15 9:15:53</p>
                    </div>
                </div>
            </div>
            <div class="info-box">
                <div class="info-header">
                    <img src="./assets/pi.png" id="pi">
                    <p>Raspberry Pi</p>
                </div>
                <div class="info-content">
                    <div class="info">
                        <p class="state">State: </p>
                        <p>Up</p>
                    </div>
                    <div class="info">
                        <p class="timestamp">Time stamp: </p>
                        <p>2025-04-15 9:15:53</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="main-wrapper" id="main-content">

    </div>
        </main>
        `;
        initFullMap();
    }

    document.querySelectorAll('.nav a').forEach(a => a.classList.remove("active"));
    document.querySelectorAll('.nav a').forEach(a => {
        if (a.textContent.toLowerCase() === view) a.classList.add("active");
    });
}


window.onload = () => loadView("home");


function initMiniMap() {
    mapboxgl.accessToken = 'pk.eyJ1IjoidGhpZW5waGF0MDgxMCIsImEiOiJjbTJzc3hrdzcwMTIwMm5weHliM2x1bXE0In0.-VE5I0KN9VglmKTWSxG97g';
    const map = new mapboxgl.Map({
        container: 'map',
        style: 'mapbox://styles/mapbox/standard',
        center: [106.8007, 10.8752],
        zoom: 15,
        pitch: 60,
        bearing: -20,
        antialias: true,
        projection: 'globe'
    });
    new mapboxgl.Marker()
        .setLngLat([106.8007, 10.8752])
        .setPopup(new mapboxgl.Popup().setHTML("<strong>Làng</strong>"))
        .addTo(map);
}

function initFullMap() {
    mapboxgl.accessToken = 'pk.eyJ1IjoidGhpZW5waGF0MDgxMCIsImEiOiJjbTJzc3hrdzcwMTIwMm5weHliM2x1bXE0In0.-VE5I0KN9VglmKTWSxG97g';
    const map = new mapboxgl.Map({
        container: 'map_2',
        style: 'mapbox://styles/mapbox/standard',
        center: [106.8007, 10.8752],
        zoom: 15,
        pitch: 60,
        bearing: -20,
        projection: 'globe',
        antialias: true
    });
    map.on('load', () => {
        import('./mapbox3d.js').then(m => {
            // Kích hoạt 3D
            // m.enable3D(map);

            // Thêm marker

            const locations = [
                {
                    popup: 'Nhà văn hóa sinh viên',
                    longitude: 106.80131197919498,
                    latitude: 10.875352088818252,
                    type: '1'
                },
                {
                    popup: 'Cổng A - Trường Đại học Công nghệ thông tin',
                    longitude: 106.80212737116507,
                    latitude: 10.870716148648393,
                    type: '0'
                },
                {
                    popup: 'Ktx Khu B - ĐHQG',
                    longitude: 106.78378106209817,
                    latitude: 10.88222157674423,
                    type: '1'
                },
                {
                    popup: 'Trường Đại học KHXH&NV  ',
                    longitude: 106.80227757476617,
                    latitude: 10.872275518446761,
                    type: '0'
                },
            ]

            locations.forEach((element, index) => {
                var center = [element.longitude, element.latitude];
                m.addCircleLayer(map, center, element.type, index);

                const mark = m.addMarker(
                    map,
                    element.popup,
                    element.longitude,
                    element.latitude,
                );
            });
        });
    });


}
