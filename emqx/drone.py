class Drone:
    def __init__(self, drone_id, drone_state, battery_level, time_stamp):
        self.drone_id = drone_id
        self.drone_state = drone_state
        self.battery_level = battery_level
        self.time_stamp = time_stamp

    def to_json(self):
        return {
            "drone_id": self.drone_id,
            "state": self.drone_state,
            "battery": self.battery_level,
            "time_stamp": self.time_stamp
        }
