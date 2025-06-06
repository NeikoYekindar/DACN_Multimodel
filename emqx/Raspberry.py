class Raspberry:
    def __init__(self, id, state, time_stamp):
        self.id = id
        self.state = state
        self.time_stamp = time_stamp

    def to_json(self):
        return {
            "id": self.id,
            "state": self.state,
            "time_stamp": self.time_stamp
        }       