const router = require("express").Router();
const verifyToken = require("../middlewares/token-verify");
const axios = require('axios');
const upload = require("../middlewares/upload");
const {Image} = require('../models');
const fs = require("fs").promises;
const fs1 =require("fs");
const FormData = require('form-data');
const {UrlIsEmpty} = require('../middlewares/Image-validator')
const multer = require("multer");
const path = require("path");

router.post("", UrlIsEmpty, upload.single('url'), async (req, res) => {
    const data = req.body;
    try {
        if (!req.file) {
            data.url = "default.jpg";
        } else {
            data.url = req.file.filename;
        }
        data.userId = req.user.id;
        const image = await Image.create(data);
        
        
        const protocol = req.protocol; 
        const host = req.get('host'); 
        const imageUrl = `${protocol}://${host}//${data.url}`;

        const flaskServerUrl = 'http://127.0.0.1:5000/predict';
        const formData = new FormData();
        formData.append('file', fs1.createReadStream(`./uploads/${data.url}`));
        formData.append('imageUrl', imageUrl);
        formData.append('userId', data.userId);

        axios.post(flaskServerUrl, formData, {
            headers: formData.getHeaders()
        })
        .then(response => {
            console.log('Flask server response:', response.data);
            res.status(201).json({
                message: `Image is uploaded. Image id: ${image.id}`,
                prediction: response.data.result
            });
        })
        .catch(error => {
            console.error('Error from Flask server:', error);
            res.status(500).json({ message: `Error processing image: ${error.message}` });
        });

    } catch (err) {
        res.status(400).json({ message: `There is a problem: ${err}` });
    }

    //     res.status(201);
    //     res.json({ message: `image is uploaded. image id: ${image.id}` });
    // } catch (err) {
    //     res.status(400);
    //     res.json({ message: `There is a problem: ${err}` });
    // }
});


router.get("",async(req,res)=>{
    const images = await Image.findAll({
        where:{
            userId: req.user.id
        },
        attributes:["id","url"]
    });
    images.map( (image)=>{
        image.url = `${req.protocol}://${req.get('host')}//${image.url}`;
      });
      console.log(images);
    
      res.status(200);
      res.json(images);
});



router.delete("/:id" ,async (req, res) => {
    const { id } = req.params;
    try {
        const image = await Image.findOne({
            where: { id: id }
        });

        if (!image) {
            res.status(404);
            res.json({ message: "Image not found" });
            return;
        }

        if (image.url != "default.jpg") {
            const filePath = `~/../uploads/${image.url}`;
            await fs.access(filePath);
            await fs.unlink(filePath);
        }
        await Image.destroy({
            where: { id: id },
        });
        res.status(200);
        res.json({ message: "Image is deleted" });
    } catch (err) {
        res.status(400);
        res.json({ message: `There is a problem: ${err}` });
    }
});



//search by item
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      const tempDir = path.join(__dirname,"..", "temp");
      if (!fs1.existsSync(tempDir)) {
        fs1.mkdirSync(tempDir);
      }
      cb(null, tempDir);
    },
    filename: (req, file, cb) => {
      cb(null, "uploaded_image" + path.extname(file.originalname)); 
    },
  });
  
  const tempupload = multer({ storage: storage });
  
  router.post("/tempupload", tempupload.single("image"), (req, res) => {
    const { label, category } = req.body;
    if (!req.file) {
      return res.status(400).send("No file uploaded.");
    }

      const protocol = req.protocol;
      const host = req.get('host');
      const imageUrl = `${protocol}://${host}//${req.file.filename}`;
      const flaskServerUrl = 'http://127.0.0.1:5000/searchByItem';
      const formData = new FormData();
      formData.append('category', category);
      formData.append('label', label);
      formData.append('image', imageUrl);
      
      axios.post(flaskServerUrl,formData)
   
    res.send({
      message: "File uploaded successfully",
      filename: req.file.filename,
      filepath: req.file.path,
      label,
      category
    });
  });



module.exports = router;