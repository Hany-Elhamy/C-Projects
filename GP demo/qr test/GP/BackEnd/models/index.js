const Sequelize = require("sequelize");
const db = require("../config/database");

const UserModel = require('./User');
const ImageModel = require('./Image');

const User = UserModel(db,Sequelize);
const Image = ImageModel(db,Sequelize);

User.hasMany(Image, {
    onDelete: "CASCADE",
    onUpdate: "CASCADE",
  }); 
Image.belongsTo(User, {
    onDelete: "CASCADE",
    onUpdate: "CASCADE",
}); 

db.sync({ force: false }).then(() => {
    console.log("Tables Created");
});

module.exports = {
    User,
    Image
}
