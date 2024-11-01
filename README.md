Project Overview: The project is a website that allows users to create a profile and upload outfits they like and receive personalized clothing recommendations based on their personal clothing style extracted by identifying certain patterns in the outfits they have listed. Users can upload outfits from real life or social media, and the system will analyze these outfits to suggest new apparel items from major fashion retailers.

Key Features:
User Profile & Outfit Upload:

Users create a profile and can upload outfit images by scanning a QR code using their mobile devices.
The uploaded images can be from any source (real life or social media) and can contain any background. The images may include both men's and women's outfits.
Users can continue adding images to their profile at any time.

Image Analysis Using Machine Learning:
When an outfit is uploaded, the first ML model segments the image to isolate individual clothing items, applying a mask to everything except the items of interest.
Each clothing item is then analyzed by a second ML model trained to recognize 17 different patterns.
Another analysis identifies the colors of the clothing items.
The processed data, including pattern, color, and user ID, is stored for further recommendations.

Clothing Database:
A web crawler continuously collects and updates clothing items from popular fashion websites (e.g., Amazon, Zara, Gucci).
These items are processed using the same ML models and stored along with their price, URL, and additional product information. This allows users to purchase matching items directly from these stores.

Recommendation Features:

Fashion Adventurer:
Users choose a category of clothing (e.g., polo shirts) and the system retrieves similar items from their uploaded outfits.
The system matches a random pattern and color from the user's clothing data and searches the database for matching items from online stores. The user is presented with a list of options and can be redirected to the store to purchase.

Fashion for Moments:
This feature offers specific clothing categories for events like weddings, sports, or formal occasions.
The system filters recommendations based on the selected category, ensuring relevance to the event or occasion.

Clone and Own:
Users can upload a single image of an outfit they like, and the system finds an exact match from the clothing database, returning the closest available options.

Your Style Signature:
The system analyzes the user’s uploaded outfits to identify their most frequently worn colors and patterns.
Based on these preferences, it recommends clothing items that match the user’s unique style, ensuring highly personalized suggestions.



KEY TECHNOLOGIES USED :

**Client-Side:**
- **Frontend Framework:** React.js
- **Styling:** CSS/Bootstrap
- **API Communication:** Axios for HTTP requests

**Server-Side:**
- **Technology Used:**
  - **Backend Framework:** Express.js (Node.js Framework)
  - **Database:** MySQL
  - **User Authentication:** JWT (JSON Web Tokens)
  - **ORM Tool:** Sequelize
  - **Web Scraping:** Python with Scrapy for web crawling
  - **API Development:** RESTful APIs for client-server communication
  - **Hosting:** AWS EC2 for server hosting, RDS for database

**Communication Between Client and Server:**
- The client and server communicate via RESTful APIs. The client makes HTTP requests (GET, POST, PUT, DELETE) to the server, which processes these requests, interacts with the database or machine learning models, and sends the appropriate responses back to the client.

**Machine Learning Models:**

**First Model:**
- **Description:** CLIP (Contrastive Language-Image Pre-training) model fine-tuned on fashion pattern recognition. It processes an image to identify patterns like gradient, floral, or zigzag, providing probabilities for each label.
- **Dataset Used:** [Fashion Pattern Images](https://huggingface.co/datasets/yainage90/fashion-pattern-images)
- **Accuracy for each pattern:**
  - Gradient: 0.90
  - Snow Flake: 0.78
  - Camouflage: 0.64
  - Dot: 0.78
  - Zebra: 0.89
  - Leopard: 0.87
  - Lettering: 0.72
  - Snake Skin: 0.84
  - Geometric: 0.74
  - Muji: 0.95
  - Floral: 0.80
  - Zigzag: 0.86
  - Graphic: 0.85
  - Paisley: 0.75
  - Tropical: 0.88
  - Checked: 0.74
  - Houndstooth: 0.86
  - Argyle: 0.64
  - Stripe: 0.88

**Second Model:**
- **Description:** The model is a pre-trained deep learning system for understanding clothing in images. It takes an image as input and predicts a detailed breakdown of what each pixel represents in terms of clothing items (hat, shirt, shoes, etc.). This is achieved by assigning a probability to each pixel belonging to a specific clothing category.
- **Dataset Used:** Human Parsing Dataset, includes detailed pixel-wise annotations for fashion images.
- **Accuracies of segmenting these items:**
  - Dress: 0.74
  - Pants: 0.90
  - Skirt: 0.76
  - Upper-clothes: 0.87
- **Overall Evaluation Metrics:**
  - Evaluation Loss: 0.15
  - Mean Accuracy: 0.80
  - Mean IoU: 0.69

**Libraries Used in the Whole Project:**
- transformers (SegformerImageProcessor, AutoModelForSemanticSegmentation, CLIPProcessor, CLIPModel)
- Image
- cv2
- requests
- torch
- torch.nn
- numpy
- csv
- os
- collections
- sklearn.cluster
- webcolors
- matplotlib.pyplot
- colorthief
- collections
