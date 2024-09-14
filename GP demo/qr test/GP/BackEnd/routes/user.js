const router = require("express").Router();
const {User} = require( "../models");


router.get('',  (req, res) => {
    res.send("welcome");
});

module.exports = router;