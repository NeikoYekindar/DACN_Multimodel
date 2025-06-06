import paho.mqtt.client as mqtt
from Station import Station
from drone import Drone
from camera import Camera
from Raspberry import Raspberry
import json

# Thiết lập client
client = mqtt.Client(clean_session=True)

# Kết nối tới MQTT broker
client.connect("54.226.9.38", 1883)

# Chạy vòng lặp MQTT trên một thread riêng
client.loop_start()

print("Publisher is running...")

try:

    camera = Camera(1, 0, "2025-04-19 9:20:53")
    drone = Drone( 1, 1, 50, "2025-04-19 9:21:53")
    raspberry = Raspberry(1, 1, "2025-04-19 9:20:53")
    station = Station("Trường Đại học KHXH&NV", "Station 3", 106.80227757476617, 10.872275518446761, camera, drone, raspberry, 0)
    
    payload = station.to_json()
    json_payload = json.dumps(payload)

    while True:
        message = input("Enter message to send (or 'exit' to quit): ")
        if message.lower() == 'exit':
            break
        client.publish("dacn/data", json_payload, qos=1)
        print(f"Sent: {json_payload}")
except KeyboardInterrupt:
    print("Stopping...")
finally:
    client.loop_stop()
    client.disconnect()
