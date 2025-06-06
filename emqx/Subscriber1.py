import paho.mqtt.client as mqtt

print("Subcriber is running...")

def on_message(client, userdata,msg):
    data = msg.payload.decode()
    print(f"{data}")

client = mqtt.Client()
client.on_message = on_message
client.connect("34.204.53.7", 1883)
client.subscribe("dacn/data", qos=1)



print("Subscriber is running...")
client.loop_forever()

