// mqttClient.js
const mqtt = require('mqtt');

const client = mqtt.connect('mqtt://54.226.9.38:1883');

function subscribe(topic, messageHandler) {
    client.on('connect', () => {
        console.log('Connected to broker');
        client.subscribe(topic, { qos: 1 }, (err) => {
            if (err) {
                console.error(`Subscribe error: ${err.message}`);
            } else {
                console.log(`Subscribed to topic: ${topic}`);
            }
        });
    });

    client.on('message', (receivedTopic, message) => {
        if (receivedTopic === topic) {
            messageHandler(receivedTopic, message.toString());
        }
    });
}

module.exports = { subscribe };
