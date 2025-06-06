const mongoose = require('mongoose')

const StationSchema = new mongoose.Schema({
    location: { type: String, required: true },
    name: { type: String, required: true },
    longitude: { type: Number, required: true },
    latitude: { type: Number, required: true },
    camera : {
        state: { type: Boolean, required: true },
        time_stamp: { type: String },
    },
    drone : {
        state: { type: Boolean, required: true },
        battery: { type: Number, required: true },
        time_stamp: { type: String },
    },
    raspberry : {
        state: { type: Boolean, required: true },
        time_stamp: { type: String },
    },
    warning: { type: Boolean, required: true }
}, { versionKey: false });

const Station = mongoose.model('Station', StationSchema)
module.exports = Station;