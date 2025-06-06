const mongoose = require('mongoose')
require('dotenv').config()

mongoose.set('strictQuery', true)

const atlat = `mongodb+srv://${process.env.usernameDB}:${process.env.passwordDB}@cluster0.dnwsy.mongodb.net/dacn?retryWrites=true&w=majority&appName=Cluster0`
const connect = async () => {
    try {
        await mongoose.connect(atlat, {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        })
        console.log("Connected success");
    } catch (error) {
        console.log("Connected fail")
        console.log(error);
    }
}
module.exports = { connect }