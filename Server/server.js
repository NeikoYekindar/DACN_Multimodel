const express = require('express')
const mongoose = require('mongoose')
const cors = require('cors')
const database = require('./database')
require('dotenv').config()
const { subscribe } = require('./Subscriber')
const Station = require('./Station')

const app = express();
app.use(express.json());

app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*'),
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE'),
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization'),
    next()
});

database.connect();
mongoose.set('strictQuery', true);

subscribe('dacn/data', async (topic, message ) => {
    console.log(`Received message on topic ${topic}: ${message}`);
    const data = JSON.parse(message);

    const existed = await Station.findOne({'name' : data.name});    

    if (!existed) {
        const newData = new Station({
            location: data.location,
            name: data.name,
            longitude: data.longitude,
            latitude: data.latitude,
            camera : {
                state: data.camera.state,
                time_stamp: data.camera.time_stamp,
            },
            drone : {
                state: data.drone.state,
                battery: data.drone.battery,
                time_stamp: data.drone.time_stamp,
            },
            raspberry : {
                state: data.raspberry.state,
                time_stamp: data.raspberry.time_stamp,
            },
            warning: data.warning
        });
    
        await newData.save()
            .then(() => {
                console.log('Data saved to MongoDB');
            })
            .catch((err) => {
                console.error('Error saving data to MongoDB:', err);
            });
    }
    else {
        existed.camera.state = data.camera.state;
        existed.camera.time_stamp = data.camera.time_stamp;
        existed.drone.state = data.drone.state;
        existed.drone.battery = data.drone.battery;
        existed.drone.time_stamp = data.drone.time_stamp;
        existed.raspberry.state = data.raspberry.state;
        existed.raspberry.time_stamp = data.raspberry.time_stamp;
        existed.warning = data.warning;

        await existed.save()
            .then(() => {
                console.log('Data updated in MongoDB');
            })
            .catch((err) => {
                console.error('Error updating data in MongoDB:', err);
            });
    }

    
})


app.get('/data', async (req, res) => {
    try {
        const data = await Station.find({});
        console.log(data);
        if (!data) {
            return res.status(404).json({ error: 'No data found' });
        }
        res.status(200).json(data)
    }
    catch (error) {
        console.error('Error fetching data from MongoDB:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});


app.listen(process.env.PORT, () => {
    console.log(`Server is running on port ${process.env.PORT}`);
})
