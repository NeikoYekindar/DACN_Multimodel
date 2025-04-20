export function addMiniMapMarker(map, location, index, iconUrl) {
    const { name, popup, longitude, latitude } = location; 

    const el = document.createElement('div');
    el.className = 'marker-mini';

    el.innerHTML = `<img src="${iconUrl}" style="width: 24px; height: 24px;">`;

    const marker = new mapboxgl.Marker(el)
        .setLngLat([longitude, latitude])
        .setPopup(
            new mapboxgl.Popup({ offset: 25 }).setHTML(`<strong>${name}</strong><br>${popup}`)
        )
        .addTo(map);

    return marker;
}
export function createCircle(center, radiusInMeters, numPoints = 64) {
    const coordinates = [];
    const angleStep = (2 * Math.PI) / numPoints;

    const radiusInDegrees = radiusInMeters / 111320;

    for (let i = 0; i < numPoints; i++) {
        const angle = i * angleStep;
        const lat = center[1] + radiusInDegrees * Math.sin(angle);
        const lon = center[0] + radiusInDegrees * Math.cos(angle);
        coordinates.push([lon, lat]);
    }

    coordinates.push(coordinates[0]); // đóng vòng

    return {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [coordinates],
        },
    };
}


export function addCircleLayer(map, center, type, idSuffix) {
    const sourceId = `circle-source-${idSuffix}`;
    const layerId = `circle-layer-${idSuffix}`;

    // Nếu layer cũ đã tồn tại, xoá để tránh lỗi
    if (map.getLayer(layerId)) {
        map.removeLayer(layerId);
    }
    if (map.getSource(sourceId)) {
        map.removeSource(sourceId);
    }

    const circleGeoJSON = createCircle(center, 100);

    map.addSource(sourceId, {
        type: 'geojson',
        data: circleGeoJSON,
    });

    const color = type == 1 ? 'rgba(253, 43, 43, 0.74)' : 'rgba(53, 247, 172, 0.88)';

    map.addLayer({
        id: layerId,
        type: 'fill',
        source: sourceId,
        paint: {
            'fill-color': color,
            'fill-outline-color': '#007bff',
            'fill-opacity': 0.5,
        },
    });
}

