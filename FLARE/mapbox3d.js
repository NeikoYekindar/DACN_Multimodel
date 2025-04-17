export function enable3D(map) {
    map.addSource('mapbox-dem', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.terrain-rgb',
        tileSize: 512,
        maxzoom: 14
    });

    map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.5 });
    map.setLight({ intensity: 0.7 });

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