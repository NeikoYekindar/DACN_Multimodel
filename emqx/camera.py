class Camera: 
    def __init__(self, camera_id, camera_state, time_stamp):
        self.camera_id = camera_id
        self.camera_state = camera_state
        self.time_stamp = time_stamp

    def to_json(self): 
        return {
            "camera_id": self.camera_id,
            "state": self.camera_state,
            "time_stamp": self.time_stamp
        }