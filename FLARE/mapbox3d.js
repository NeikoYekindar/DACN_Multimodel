function enable3D(map) {
    map.addSource('mapbox-dem', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.terrain-rgb',
        tileSize: 512,
        maxzoom: 14
    });

    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });
    map.setLights({ intensity: 0.7 });

    map.addLayer({
        id: '3d-buildings',
        source: 'composite',
        'source-layer': 'building',
        filter: ['==', 'extrude', 'true'],
        type: 'fill-extrusion',
        minzoom: 15,
        paint: {
            'fill-extrusion-color': [
                'interpolate',
                ['linear'],
                ['zoom'],
                15, '#e0e0e0',
                16, '#f5f5f5'
            ],
            'fill-extrusion-height': ['get', 'height'],
            'fill-extrusion-base': ['get', 'min_height'],
            'fill-extrusion-opacity': 0.85
        }
    });
}

//Thêm marker vào mapmap
function addMarker(map, name, popupHtml = '', longitude, latitude, { camera_state, drone_state, pi_state, camera_timestamp, drone_timestamp, pi_timestamp, battery_drone, warning }) {

    const customIcon = document.createElement('div');
    customIcon.className = 'custom-marker';
    customIcon.style.backgroundImage = 'url(./assets/drone.png)';
    customIcon.style.width = '32px';
    customIcon.style.height = '32px';
    customIcon.style.backgroundSize = '100%';

    const marker = new mapboxgl.Marker(customIcon)
        .setLngLat([longitude, latitude]);

    if (popupHtml) {
        const popup = new mapboxgl.Popup().setHTML(popupHtml);
        marker.setPopup(popup);
    }

    const popup = document.getElementById('station_detail');

    marker.getElement().addEventListener('click', () => {

        const updateAndShowPopup = () => {
            const popup_station_name = document.getElementById('station_name');

            const popupCameraState = document.getElementById('state_camera');
            const popupDroneState = document.getElementById('state_drone');
            const popupPiState = document.getElementById('state_pi');
            const popupCameraTimestamp = document.getElementById('timestamp_camera');
            const popupDroneTimestamp = document.getElementById('timestamp_drone');
            const popupPiTimestamp = document.getElementById('timestamp_pi');
            const popupPiBattery = document.getElementById('battery_drone');
            const popupType = document.getElementById('type');

            popup_station_name.innerHTML = name;
            popupCameraState.innerHTML = camera_state;
            popupDroneState.innerHTML = drone_state;
            popupPiState.innerHTML = pi_state;
            popupCameraTimestamp.innerHTML = camera_timestamp;
            popupDroneTimestamp.innerHTML = drone_timestamp;
            popupPiTimestamp.innerHTML = pi_timestamp;
            popupPiBattery.innerHTML = battery_drone;
            popupType.innerHTML = warning ? 'Flood risk' : 'Normal';
            popupType.style.backgroundColor = warning ? 'red' : 'green';
            popupType.style.borderColor = warning ? 'red' : 'green';
            popupType.style.color = 'white';

            popup.style.display = 'flex';
            requestAnimationFrame(() => {
                popup.classList.add('show');
            });
        }

        if (popup.classList.contains('show')) {
            popup.classList.remove('show');
            setTimeout(() => {
                popup.style.display = 'none';
                updateAndShowPopup();
            }, 300); // chờ animation ẩn xong
        } else {
            updateAndShowPopup();
        }
    });

    document.getElementById("btn_back").addEventListener("click", () => {
        // document.getElementById("station_detail").style.display = 'none';
        popup.classList.remove('show');
        setTimeout(() => {
            popup.style.display = 'none';
        }, 300); // chờ animation kết thúc
    });
    marker.addTo(map);
    return marker;
}

//Tạo circle xung quanh markermarker
function createCircle(center, radiusInMeters, numPoints = 64) {
    const coordinates = [];
    const angleStep = (2 * Math.PI) / numPoints;

    // Radius của 1 độ (khoảng 111,32 km) tính theo bán kính của Trái Đất
    const radiusInDegrees = radiusInMeters / 111320;

    for (let i = 0; i < numPoints; i++) {
        const angle = i * angleStep;
        const lat = center[1] + radiusInDegrees * Math.sin(angle);
        const lon = center[0] + radiusInDegrees * Math.cos(angle);
        coordinates.push([lon, lat]);
    }

    coordinates.push(coordinates[0]); // Đóng vòng tròn

    return {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [coordinates]
        }
    };
}

//Thêm circle vừa tạo vào map
function addCircleLayer(map, center, type, idSuffix) {
    const sourceId = `circle-source-${idSuffix}`;
    const layerId = `circle-layer-${idSuffix}`;

    if (map.getSource(sourceId)) {
        map.removeLayer(layerId);
        map.removeSource(sourceId);
    }

    // Tạo circle dưới dạng GeoJSON
    const circleGeoJSON = createCircle(center, 100);

    // Thêm source và layer mới vào bản đồ
    map.addSource(sourceId, {
        type: 'geojson',
        data: circleGeoJSON
    });

    const color = (type == 1) ? 'rgba(253, 43, 43, 0.74)' : 'rgba(53, 247, 172, 0.88)';

    map.addLayer({
        id: layerId,
        type: 'fill',
        source: sourceId,
        paint: {
            'fill-color': `${color}`, // Màu vòng tròn
            'fill-outline-color': '#007bff', // Màu viền
            'fill-opacity': 0.5 // Độ mờ
        }
    });
}

export { enable3D, addMarker, addCircleLayer, createCircle };