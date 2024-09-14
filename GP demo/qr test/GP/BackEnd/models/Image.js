module.exports = (db, type) =>{
    return db.define("Image",{
        id: {
            type: type.INTEGER,
            autoIncrement: true,
            primaryKey: true,
        },
        url:{
            type: type.STRING,
            allowNull: false,
        },
    });
}