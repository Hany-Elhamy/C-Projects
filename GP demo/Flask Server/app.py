from flask import Flask, request, jsonify
import csv

from flask_cors import CORS
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch.nn as nn
import numpy as np
import os
from collections import Counter
from sklearn.cluster import KMeans
import webcolors
import matplotlib.pyplot as plt
from colorthief import ColorThief
import re
import torch
import requests
from io import BytesIO
import random
import os
import pandas as pd
import csv
from collections import Counter
import time
x = []
y = []
z = []
import csv
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation, CLIPProcessor, CLIPModel
from PIL import Image
import cv2
import torch.nn as nn
import numpy as np
import os
from collections import Counter
import re
import torch
from sklearn.cluster import KMeans
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import requests
import torch.nn as n

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/predict": {"origins": "http://localhost:3000"}})

@app.route('/')
def home():
    return "Welcome to the Flask server!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    user_id = request.form.get('userId')
    image_url = request.form.get('imageUrl')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    print(f"Received file: {file.filename}")
    print(f"Received userId: {user_id}")
    print(f"Received imageUrl: {image_url}")

    if file:
        # Process the file (e.g., run your AI model on the file)

    # region Processing Code
        if not hasattr(np, 'asscalar'):
            def asscalar(a):
                return a.item()

            np.asscalar = asscalar

        def load_color_dict(csv_file):
            color_dict = {}
            with open(csv_file, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if row:
                        hex_color = row[1].lstrip('#')  # Assuming "hex" is in the second column
                        primary_color = row[2]  # Assuming "primary color" is in the third column
                        color_dict[hex_color] = primary_color
            return color_dict

        color_dict = load_color_dict("C:/Users/AHMED SaYED/Desktop/Flask Server/colors_3.csv")

        def RGB2HEX(color):
            return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

        def HEX2RGB(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

        def closest_color(rgb_color):
            min_distance = float('inf')
            closest_primary_color = None
            color_obj = sRGBColor(*rgb_color, is_upscaled=True)
            lab_color = convert_color(color_obj, LabColor)
            for hex_color, primary_color in color_dict.items():
                color_obj = sRGBColor(*HEX2RGB(hex_color), is_upscaled=True)
                lab_color_dict = convert_color(color_obj, LabColor)
                distance = delta_e_cie2000(lab_color, lab_color_dict)
                if distance < min_distance:
                    min_distance = distance
                    closest_primary_color = primary_color
            return closest_primary_color

        def get_image(image_path):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        def get_colors(image, number_of_colors, min_area_ratio=0.1):
            pixels = image.reshape(-1, 3)
            pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

            if len(pixels) == 0:
                print(f"No significant pixels found in image.")
                return []

            kmeans = KMeans(n_clusters=number_of_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            labels = kmeans.labels_

            label_counts = np.bincount(labels)
            total_count = len(labels)
            color_ratios = label_counts / total_count
            significant_colors = [colors[i] for i in range(len(colors)) if color_ratios[i] >= min_area_ratio]

            sorted_colors = sorted(zip(color_ratios, significant_colors), reverse=True, key=lambda x: x[0])
            top_two_colors = [color for _, color in sorted_colors[:2]]

            return top_two_colors

        def download_image(image_url):
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    return image
                else:
                    print(f"Failed to download image. Status code: {response.status_code}")
                    return None
            except Exception as e:
                print(f"Error downloading image: {e}")
                return None

        def segment_clothes(image):
            lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            pixels = lab_image.reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=0)
            kmeans.fit(pixels)
            centers = kmeans.cluster_centers_
            gray_cluster_index = np.argmin(np.linalg.norm(centers - [128, 128, 128], axis=1))
            mask = (kmeans.labels_ != gray_cluster_index).reshape(image.shape[:2])
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            return mask

        def apply_mask(image, mask):
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            return masked_image

        def segment_and_display(image, output_dir):
            processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            inputs = processor(images=image, return_tensors="pt")

            outputs = model(**inputs)
            logits = outputs.logits.cpu()

            upsampled_logits = nn.functional.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )

            for label in range(4, 8):
                pred_seg = upsampled_logits.argmax(dim=1)[0]
                segmentation_mask_np = pred_seg.numpy()
                label_mask = np.where(segmentation_mask_np == label, 255, 0).astype(np.uint8)

                if np.any(label_mask):
                    extracted_item_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGRA)
                    extracted_item_cv2[np.where(label_mask == 0)] = [0, 0, 0, 255]
                    output_path = os.path.join(output_dir, f"segmented_label_{label}.png")
                    extracted_item = Image.fromarray(cv2.cvtColor(extracted_item_cv2, cv2.COLOR_BGRA2RGBA))
                    extracted_item.save(output_path)

        def process_image(image_path):
            image = Image.open(image_path)
            inputs = processor(
                text=labels,
                images=image,
                padding=True,
                return_tensors="pt",
                truncation=True
            )
            with torch.no_grad():
                outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).squeeze()
            filtered_patterns = [(labels[i], probs[i].item()) for i in range(len(labels)) if probs[i].item() > 0.25]
            return filtered_patterns

        def append_to_csv(row, csv_file):
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

        output_csv = r"C:\Users\AHMED SaYED\Desktop\pattcolor.csv"
        if not os.path.isfile(output_csv):
            with open(output_csv, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["userID", "itemLabel", "imagePath", "Pattern", "Color"])

        processed_images = set()
        if os.path.isfile(output_csv):
            with open(output_csv, mode='r') as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    processed_images.add(row[2])

        ckpt = "yainage90/fashion-pattern-clip"
        processor = CLIPProcessor.from_pretrained(ckpt)
        model = CLIPModel.from_pretrained(ckpt)

        labels = [
            "gradient", "snow_flake", "camouflage", "dot", "zebra", "leopard",
            "lettering", "snake_skin", "geometric", "muji", "floral", "zigzag",
            "graphic", "paisley", "tropical", "checked", "houndstooth", "argyle",
            "stripe"
        ]

        output_dir = "segmented6_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def process_image_url(image_url, user_id):
            image = download_image(image_url)
            if image is None:
                return
            temp_image_path = 'temp_image_from_url.jpg'
            image.save(temp_image_path)
            segment_and_display(image, output_dir)

            for segmented_filename in os.listdir(output_dir):
                if segmented_filename.endswith(('.png', '.jpg', '.jpeg')):
                    segmented_image_path = os.path.join(output_dir, segmented_filename)
                    match = re.search(r'segmented_label_(\d+)', segmented_filename)
                    if match:
                        item_label = match.group(1)
                    else:
                        item_label = "unknown"

                    patterns = process_image(segmented_image_path)
                    image = get_image(segmented_image_path)
                    rgb_colors = get_colors(image, 5, min_area_ratio=0.1)
                    primary_colors = [closest_color(color) for color in rgb_colors]
                    pattern_names = ", ".join([pattern[0] for pattern in patterns])
                    color_names = ", ".join(primary_colors)

                    row = [user_id, item_label, image_url, pattern_names, color_names]
                    append_to_csv(row, output_csv)

            for file in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, file))

        # Replace with your image URL and user ID
        image_url = image_url
        user_id = user_id
        process_image_url(image_url, user_id)

        print("Done")

        # endregion

        result = "dummy_result"
        return jsonify({'result': result, 'userId': user_id, 'imageUrl': image_url}), 200

@app.route('/searchcode', methods=['POST'])
def search_code():
    search_code_userId = request.form.get('search_code_userId')
    search_code_label = request.form.get('itemLabel')
    search_code_category = request.form.get('category')

    print(f"Received search_code_userId: {search_code_userId}")
    print(f"Received search_code_label: {search_code_label}")
    print(f"Received search_code_category: {search_code_category}")
    x.clear()

    # region search_code
    # Primary

    # Function to get a random pattern and color from a DataFrame
    def random_pattern_and_color(df):
        # Drop rows where 'Pattern' or 'Color' is NaN
        df = df.dropna(subset=['Pattern', 'Color'])

        # Ensure there are non-empty values in 'Pattern' and 'Color' columns
        if not df.empty:
            randomPattern = random.choice(df['Pattern'].tolist())
            randomColor = random.choice(df['Color'].tolist())
            return randomPattern, randomColor
        else:
            return None, None

    # Function to search for matching rows in a CSV file
    def search_csv_for_combination(csv_file_path, colors, randomPattern):
        matching_rows = []

        color_list = colors.split(', ')

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                row_colors = row.get('colors', '').split(', ')
                row_patterns = row.get('patterns', '').split(', ')

                if all(color in row_colors for color in color_list) and randomPattern in row_patterns:
                    matching_rows.append(row)

        return matching_rows

    # Paths to the CSV files
    csv_file_path_patterns = r"C:\Users\AHMED SaYED\Desktop\pattcolor.csv"
    csv_file_path_search = search_code_category

    user_id = int(search_code_userId)
    item_label = int(search_code_label)

    try:
        # Read the DataFrame from the CSV file
        df = pd.read_csv(csv_file_path_patterns)

        # Filter the DataFrame based on user_id and item_label

        filtered_df = df[(df['userID'] == user_id) & (df['itemLabel'] == item_label)]

        # print(filtered_df)

        matching_rows = []


        while not matching_rows:
            # Get a random pattern and color
            randomPattern, randomColor = random_pattern_and_color(filtered_df)

            if randomPattern and randomColor:
                # Search for matching rows in the other CSV file
                matching_rows = search_csv_for_combination(csv_file_path_search, randomColor, randomPattern)

                if not matching_rows:
                    print(f"No match found for Pattern: {randomPattern} and Color: {randomColor}. Retrying...")
            else:
                print("No valid pattern or color found. Exiting...")
                break

        if matching_rows:
            # Print matching rows
            print(f"Chosen Pattern: {randomPattern} and Color: {randomColor}.")
            for row in matching_rows:
                x.append(row)
                print(row)
        else:
            print("No matching rows found after several attempts.")

    except Exception as e:
        print(f"An error occurred: {e}")
    # endregion

    if not search_code_userId or not search_code_label or not search_code_category:
        return jsonify({'error': 'Missing data'}), 400

    result = "dummy_result"
    return jsonify({'label': search_code_label, 'userId': search_code_userId, 'imageUrl': search_code_category}), 200

@app.route('/onetimeshot', methods=['POST'])
def one_time_shot():
    one_time_shot_userId = request.form.get('one_time_shot_userId')
    one_time_shot_label = request.form.get('oneTimeShotLabel')
    one_time_shot_category = request.form.get('oneTimeShotcategory')

    print(f"Received one_time_shot_userId: {one_time_shot_userId}")
    print(f"Received one_time_shot_label: {one_time_shot_label}")
    print(f"Received one_time_shot_category: {one_time_shot_category}")
    y.clear()

    # region one_time_shot
    # Function to count and get the most common item in a list
    def get_most_common(lst):
        if lst:
            return Counter(lst).most_common(1)[0][0]
        return None

    # Function to search for matching rows in a CSV file
    def search_csv_for_combination(csv_file_path, color, pattern):
        matching_rows = []

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                row_colors = row.get('colors', '').split(', ')
                row_patterns = row.get('patterns', '').split(', ')

                if color in row_colors and pattern in row_patterns:
                    matching_rows.append(row)

        return matching_rows

    # Paths to the CSV files
    csv_file_path_patterns = r"C:\Users\AHMED SaYED\Desktop\pattcolor.csv"
    csv_file_path_search = one_time_shot_category

    try:
        # Read the DataFrame from the CSV file
        df = pd.read_csv(csv_file_path_patterns)

        # Filter the DataFrame based on user_id and item_label
        user_id = int(one_time_shot_userId)
        item_label = int(one_time_shot_label)
        filtered_df = df[(df['userID'] == user_id) & (df['itemLabel'] == item_label)]

        if not filtered_df.empty:
            # Count the occurrences of each color and pattern
            colors = filtered_df['Color'].dropna().tolist()
            patterns = filtered_df['Pattern'].dropna().tolist()

            # Get the most common color and pattern
            most_common_color = get_most_common(colors)
            most_common_pattern = get_most_common(patterns)

            print("Most Common Color:", most_common_color)
            print("Most Common Pattern:", most_common_pattern)

            if most_common_color and most_common_pattern:
                # Search for matching rows in the other CSV file
                matching_rows = search_csv_for_combination(csv_file_path_search, most_common_color, most_common_pattern)

                # Check if any matching rows were found
                if matching_rows:
                    # Print matching rows
                    for row in matching_rows:
                        y.append(row)
                        print(row)
                else:
                    print("No matching rows found in the CSV file for the most common color and pattern.")
            else:
                print("No valid most common color or pattern found.")
        else:
            print("No data found for the given user_id and item_label.")

    except Exception as e:
        print(f"An error occurred: {e}")
    # endregion

    if not one_time_shot_userId or not one_time_shot_label or not one_time_shot_category:
        return jsonify({'error': 'Missing data'}), 400

    result = "dummy_result"
    return jsonify({'label': one_time_shot_label, 'userId': one_time_shot_userId, 'imageUrl': one_time_shot_category}), 200

@app.route('/occasion', methods=['POST'])
def occasion():
    ocation_userId = request.form.get('ocation_userId')
    ocation_label = request.form.get('ocationLabel')
    ocation_category = request.form.get('ocationcategory')

    print(f"Received occasion_userId: {ocation_userId}")
    print(f"Received occasion_label: {ocation_label}")
    print(f"Received occasion_category: {ocation_category}")
    z.clear()

    # region occasion
    # Primary

    # Function to get a random pattern and color from a DataFrame
    def random_pattern_and_color(df):
        # Drop rows where 'Pattern' or 'Color' is NaN
        df = df.dropna(subset=['Pattern', 'Color'])

        # Ensure there are non-empty values in 'Pattern' and 'Color' columns
        if not df.empty:
            randomPattern = random.choice(df['Pattern'].tolist())
            randomColor = random.choice(df['Color'].tolist())
            return randomPattern, randomColor
        else:
            return None, None

    # Function to search for matching rows in a CSV file
    def search_csv_for_combination(csv_file_path, colors, randomPattern):
        matching_rows = []

        color_list = colors.split(', ')

        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                row_colors = row.get('colors', '').split(', ')
                row_patterns = row.get('patterns', '').split(', ')

                if all(color in row_colors for color in color_list) and randomPattern in row_patterns:
                    matching_rows.append(row)

        return matching_rows

    # Paths to the CSV files
    csv_file_path_patterns = r"C:\Users\AHMED SaYED\Desktop\pattcolor.csv"
    csv_file_path_search = ocation_category

    user_id = int(ocation_userId)
    item_label = int(ocation_label)

    try:
        # Read the DataFrame from the CSV file
        df = pd.read_csv(csv_file_path_patterns)

        # Filter the DataFrame based on user_id and item_label

        filtered_df = df[(df['userID'] == user_id) & (df['itemLabel'] == item_label)]

        # print(filtered_df)

        matching_rows = []

        while not matching_rows:
            # Get a random pattern and color
            randomPattern, randomColor = random_pattern_and_color(filtered_df)

            if randomPattern and randomColor:
                # Search for matching rows in the other CSV file
                matching_rows = search_csv_for_combination(csv_file_path_search, randomColor, randomPattern)

                if not matching_rows:
                    print(f"No match found for Pattern: {randomPattern} and Color: {randomColor}. Retrying...")
            else:
                print("No valid pattern or color found. Exiting...")
                break

        if matching_rows:
            # Print matching rows
            print(f"Chosen Pattern: {randomPattern} and Color: {randomColor}.")
            for row in matching_rows:
                z.append(row)
                print(row)
        else:
            print("No matching rows found after several attempts.")

    except Exception as e:
        print(f"An error occurred: {e}")
    # endregion

    if not ocation_userId or not ocation_label or not ocation_category:
        return jsonify({'error': 'Missing data'}), 400

    result = "dummy_result"
    return jsonify({'label': ocation_label, 'userId': ocation_userId, 'imageUrl': ocation_category}), 200

@app.route('/searchByItem', methods=['POST'])
def searchByItem():
    searchByItem_image = request.form.get('image')
    searchByItem_label = request.form.get('label')
    searchByItem_category = request.form.get('category')

    print(f"Received searchByItem_image: {searchByItem_image}")
    print(f"Received occasion_label: {searchByItem_label}")
    print(f"Received occasion_category: {searchByItem_category}")

    if not searchByItem_image or not searchByItem_label or not searchByItem_category:
        return jsonify({'error': 'Missing data'}), 400

    result = "dummy_result"
    return jsonify({'label': searchByItem_label, 'image': searchByItem_image, 'category': searchByItem_category}), 200

@app.route('/searchcode', methods=['GET'])
def search_code_get():
    time.sleep(1)
    data = x
    return jsonify(data)

@app.route('/onetimeshot', methods=['GET'])
def one_time_shot_get():
    time.sleep(1)
    data = y
    return jsonify(data)

@app.route('/occasion', methods=['GET'])
def occasion_get():
    time.sleep(1)
    return jsonify(z)


if __name__ == "__main__":
    app.run(debug=True)

