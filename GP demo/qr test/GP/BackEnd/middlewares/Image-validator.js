const { body } = require('express-validator');

const ImageUrl = body("url").notEmpty().withMessage("Image cannot be empty");

module.exports = {
    UrlIsEmpty : ImageUrl,

};