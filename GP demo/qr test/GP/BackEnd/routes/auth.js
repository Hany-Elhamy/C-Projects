const router = require("express").Router();
const {User} = require( "../models");
const { isEmail, isPassword, isName, isGender } = require('../middlewares/auth-validator');
const { validationResult } = require('express-validator');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const qr = require('qrcode');
const fs = require('fs');
const axios = require('axios');

router.post('/register', isEmail, isPassword, isName, isGender, async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            res.status(400);
            return res.json({
                message: errors.array()
            });
        }

        const data = req.body;
        const user = await User.findOne({
            where: { email: data.email },
        });

        if (user !== null) {
            res.status(400);
            return res.json({ message: "This Email is already exist" });
        }

        const saltRounds = 10;
        const salt = await bcrypt.genSalt(saltRounds);
        const hashedPassword = await bcrypt.hash(data.password, salt);
        data.password = hashedPassword;

        var _data = await User.create(data);
        console.log(_data.dataValues);
        var qrCode = generateQRCode(_data.email,_data.password,_data.id,_data.name)
        console.log(data);

        res.status(201);
        res.json({
            message: "User created successfully",
            qrCode: qrCode
        });

    } catch (err) {
        res.status(400);
        res.json({ message: `There is a problem: ${err}` });
    }
});




router.post("/login", isEmail, isPassword, async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            res.status(400);
            return res.json({
                message: errors.array()
            });
        }

        const data = req.body;
        const user = await User.findOne({
            where: { email: data.email },
        });

        if (user === null) {
            res.status(400);
            return res.json({ message: "Email or Password Not Found" });
        }

        if (!(await bcrypt.compare(data.password, user.password))) {
            res.status(400);
            return res.json({ message: "Email or passord not found" });
        }

        const token = generateAccessToken({
            name: user.name,
            email: user.email,
            id: user.id
        });

        res.status(200);
        res.json({ token: token });
    } catch (err) {
        res.status(400);
        res.json({ message: `There is a problem: ${err}` });
    }
});


const generateAccessToken = (userData) => {
    return jwt.sign(userData, process.env.TOKEN_SECRET);
};


const generateQRCode = (userEmail,userPassword,userId,UserName) => {
    var token = generateAccessToken({
        name: UserName,
        email: userEmail,
        password: userPassword,
        id: userId
    });
    const link = `http://localhost:3000/profile?token=${token}`;

    const filename = `C:/Users/AHMED SaYED/Desktop/qr test/GP/BackEnd/qr-codes/${userEmail}.png`; 

    qr.toFile(
      filename,
      link,
      {
        color: {
          dark: "#000", 
          light: "#fff", 
        },
      },
      (err) => {
        if (err) throw err;
        console.log("QR code saved as", filename);
      }
    );


    return `http://localhost:4000/qr-codes/${userEmail}.png`;
  };





module.exports = router;
