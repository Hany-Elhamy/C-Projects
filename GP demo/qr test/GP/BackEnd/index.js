const express = require("express");
const path = require('path');
const bodyParser = require("body-parser");
const cors = require("cors");

const userRoutes = require('./routes/user');
const imageRoutes = require('./routes/image');
const authRoutes = require('./routes/auth')
const verifyToken = require("./middlewares/token-verify");

const app = express();
const port = process.env.PORT || 4000;
const host = process.env.HOST || "localhost";

app.use(cors());

app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(express.static("uploads"));
app.use(express.static("qr-codes"));
app.use(express.static("temp"));


const db = require("./config/database");

app.use('/api/user', userRoutes);
app.use("/api/image",verifyToken, imageRoutes);
app.use('/api/auth', authRoutes);


app.listen(port, host, () => {
    console.log("Server is running on port",port);
});

