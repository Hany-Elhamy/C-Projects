const { body } = require('express-validator');

const emailValidation = body("email")
    .isEmail()
    .withMessage("Invalid Email");

const passwordValidation = body("password")
    .isLength({ min: 3, max: 30 })
    .withMessage("Password should be between 3-30");

const nameValidation = body("name")
    .isString()
    .notEmpty()
    .withMessage("Invalid Name");

const genderValidator = body("gender")
    .isIn(["Male", "Female"])
    .withMessage("Invalid Gender");


module.exports = {
    isEmail: emailValidation,
    isPassword: passwordValidation,
    isName: nameValidation,
    isGender: genderValidator
};
