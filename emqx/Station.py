class Station:
    def __init__(self, location, name, longitude, latitude, camera, drone, raspberry, warning): 
        self.location = location
        self.name = name
        self.longitude = longitude
        self.latitude = latitude
        self.camera = camera
        self.drone = drone
        self.raspberry = raspberry
        self.warning = warning
    
    def to_json(self):
        return {
            "location": self.location,
            "name": self.name,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "camera": self.camera.to_json(),
            "drone": self.drone.to_json(),
            "raspberry": self.raspberry.to_json(),
            "warning": self.warning
        }