import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
import pandas as pd
from flask import Flask, request, render_template, redirect
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
import os
os.sys.path
import os.path
import csv
from csv import reader
import numpy as np


import pandas as pd
import urllib
import urllib.request
import urllib.request  as urllib2
from skimage import io
import random


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load the model from the .pth file
def load_model(model_weights_path, num_classes):
    model = models.efficientnet_b4(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    return model

# Function to preprocess the input image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# Function to predict the image class
def predict_image_class(model, input_tensor, class_names):
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

# Function to extract skin from an image
def extractSkin(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)

# Function to remove black from clusters
def removeBlack(estimator_labels, estimator_cluster):
    hasBlack = False
    occurance_counter = Counter(estimator_labels)
    compare = lambda x, y: Counter(x) == Counter(y)
    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]
        if compare(color, [0, 0, 0]) == True:
            del occurance_counter[x[0]]
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break
    return (occurance_counter, estimator_cluster, hasBlack)

# Function to get color information
def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
    occurance_counter = None
    colorInformation = []
    hasBlack = False

    if hasThresholding:
        occurance, cluster, black = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)

    totalOccurance = sum(occurance_counter.values())

    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = int(x[0])
        index = (index - 1) if (hasThresholding and hasBlack and int(index) != 0) else index
        color = estimator_cluster[index].tolist()
        color_percentage = x[1] / totalOccurance
        colorInfo = {"cluster_index": index, "color": color, "color_percentage": color_percentage}
        colorInformation.append(colorInfo)

    return colorInformation

# Function to extract dominant color
def extractDominantColor(image, number_of_colors=5, hasThresholding=False):
    if hasThresholding:
        number_of_colors += 1

    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1]), 3)
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)
    estimator.fit(img)
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation

# Function to plot color bar
def plotColorBar(colorInformation):
    color_bar = np.zeros((100, 700, 3), dtype="uint8")
    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])
        color = tuple(map(int, (x['color'])))
        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar

# Function to print data
def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

# Function to predict outfit recommendations based on CSV
def get_recommendations(csv_path, predicted_class, skin_tone, body_shape):
    df = pd.read_csv(csv_path)
    recommendations = df[(df['Face Shape'] == predicted_class) & (df['Skin Color'] == skin_tone) & (df['Body Shape'] == body_shape)]
    if recommendations.empty:
        return None
    return recommendations

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        body_shape = request.form['body_shape']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        # Load the model
        model_weights_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion\\best_model.pth"
        num_classes = 5
        class_names = ["Heart", "Oblong", "Oval", "Round", "Square"]
        model = load_model(model_weights_path, num_classes)

        # Preprocess the input image
        input_tensor = preprocess_image(image_path)

        # Predict the image class
        predicted_class = predict_image_class(model, input_tensor, class_names)

        # Open a simple image for skin detection and color extraction
        img = cv2.imread(image_path)
        img = imutils.resize(img, width=250)

        skin = extractSkin(img)
        dominantColors = extractDominantColor(skin, hasThresholding=True)

        color_value = [value['color'] for value in dominantColors]

        if (color_value[0][0] > 180 and color_value[0][1] > 130 and color_value[0][2] > 100) and (
                color_value[1][0] > 180 and color_value[1][1] > 130 and color_value[1][2] > 100):
            skin_tone = "Fair"
        elif (color_value[0][0] > 160 and color_value[0][1] > 100 and color_value[0][2] > 60) and (
                color_value[1][0] > 160 and color_value[1][1] > 100 and color_value[1][2] > 60):
            skin_tone = "Wheatish"
        elif (color_value[0][0] > 160 and color_value[0][1] > 80 and color_value[0][2] > 40) and (
                color_value[1][0] > 160 and color_value[1][1] > 80 and color_value[1][2] > 40):
            skin_tone = "Medium Brown"
        elif (color_value[0][0] > 130 and color_value[0][1] > 70 and color_value[0][2] > 60) and (
                color_value[1][0] > 130 and color_value[1][1] > 70 and color_value[1][2] > 60):
            skin_tone = "Brown"
        else:
            skin_tone = "Dark Brown"
            
            
        
        if skin_tone in ["Fair", "Wheatish", "Medium Brown", "Brown", "Dark Brown"]:
            dress_colors = {
                "Fair": [
                     {"color_percentage": 0.0, "color": (173, 216, 230)},     # Light Blue
                        {"color_percentage": 0.1, "color": (245, 245, 220)},     # Beige
                        {"color_percentage": 0.2, "color": (0, 0, 128)},         # Navy
                        {"color_percentage": 0.3, "color": (128, 128, 0)},       # Olive Green
                        {"color_percentage": 0.4, "color": (128, 0, 0)},         # Maroon
                        {"color_percentage": 0.5, "color": (255, 253, 208)},     # Cream
                        {"color_percentage": 0.6, "color": (101, 67, 33)},       # Dark Brown
                        {"color_percentage": 0.7, "color": (255, 219, 88)},      # Mustard
                        {"color_percentage": 0.8, "color": (0, 128, 128)},       # Teal
                        {"color_percentage": 0.9, "color": (0, 0, 0)},           # Black
                        {"color_percentage": 1.0, "color": (128, 128, 128)},     # Grey
                        {"color_percentage": 1.1, "color": (128, 0, 32)},        # Burgundy
                        {"color_percentage": 1.2, "color": (0, 0, 128)},         # Navy Blue
                        {"color_percentage": 1.3, "color": (34, 139, 34)},       # Forest Green
                        {"color_percentage": 1.4, "color": (54, 69, 79)}
                ],
                "Wheatish": [
                    {"color_percentage": 0.0, "color": (173, 216, 230)},     # Light Blue
                    {"color_percentage": 0.1, "color": (0, 0, 128)},         # Navy
                    {"color_percentage": 0.2, "color": (245, 245, 220)},     # Beige
                    {"color_percentage": 0.3, "color": (128, 128, 0)},       # Olive Green
                    {"color_percentage": 0.4, "color": (128, 0, 0)},         # Maroon
                    {"color_percentage": 0.5, "color": (255, 253, 208)},     # Cream
                    {"color_percentage": 0.6, "color": (101, 67, 33)},       # Dark Brown
                    {"color_percentage": 0.7, "color": (255, 219, 88)},      # Mustard
                    {"color_percentage": 0.8, "color": (0, 128, 128)},       # Teal
                    {"color_percentage": 0.9, "color": (0, 0, 0)},           # Black
                    {"color_percentage": 1.0, "color": (128, 128, 128)}      # Grey
                ],
                "Medium Brown": [
                    {"color_percentage": 0.0, "color": (107, 142, 35)},      # Olive Green
                    {"color_percentage": 0.1, "color": (128, 0, 0)},         # Maroon
                    {"color_percentage": 0.2, "color": (255, 253, 208)},     # Cream
                    {"color_percentage": 0.3, "color": (101, 67, 33)},       # Dark Brown
                    {"color_percentage": 0.4, "color": (255, 219, 88)},      # Mustard
                    {"color_percentage": 0.5, "color": (0, 128, 128)},       # Teal
                    {"color_percentage": 0.6, "color": (0, 0, 0)},           # Black
                    {"color_percentage": 0.7, "color": (128, 128, 128)},     # Grey
                    {"color_percentage": 0.8, "color": (128, 0, 32)},        # Burgundy
                    {"color_percentage": 0.9, "color": (0, 0, 128)},         # Navy Blue
                    {"color_percentage": 1.0, "color": (54, 69, 79)}         # Charcoal
                ],
                "Brown": [
                    {"color_percentage": 0.0, "color": (173, 216, 230)},     # Light Blue
                    {"color_percentage": 0.1, "color": (0, 0, 128)},         # Navy
                    {"color_percentage": 0.2, "color": (245, 245, 220)},     # Beige
                    {"color_percentage": 0.3, "color": (107, 142, 35)},      # Olive Green
                    {"color_percentage": 0.4, "color": (128, 0, 0)},         # Maroon
                    {"color_percentage": 0.5, "color": (255, 253, 208)},     # Cream
                    {"color_percentage": 0.6, "color": (101, 67, 33)},       # Dark Brown
                    {"color_percentage": 0.7, "color": (255, 219, 88)},      # Mustard
                    {"color_percentage": 0.8, "color": (0, 128, 128)},       # Teal
                    {"color_percentage": 0.9, "color": (0, 0, 0)},           # Black
                    {"color_percentage": 1.0, "color": (128, 128, 128)}      # Grey
                ],
                "Dark Brown": [
                    {"color_percentage": 0.1, "color": (101, 67, 33)},       # Dark Brown
                    {"color_percentage": 0.1, "color": (255, 219, 88)},      # Mustard
                    {"color_percentage": 0.2, "color": (0, 128, 128)},       # Teal
                    {"color_percentage": 0.3, "color": (0, 0, 0)},           # Black
                    {"color_percentage": 0.4, "color": (128, 128, 128)},     # Grey
                    {"color_percentage": 0.5, "color": (128, 0, 32)},        # Burgundy
                    {"color_percentage": 0.5, "color": (0, 0, 128)},         # Navy Blue
                    {"color_percentage": 0.6, "color": (34, 139, 34)},       # Forest Green
                    {"color_percentage": 0.7, "color": (54, 69, 79)}         # Charcoal
                ],
        }
        dress_color = dress_colors[skin_tone]

        def plotColorSquares(colors):
            num_colors = len(colors)
            fig, ax = plt.subplots(1, num_colors, figsize=(num_colors, 2))

            for i, color in enumerate(colors):
                ax[i].imshow([[color['color']]])
                ax[i].axis('off')

            plt.savefig('static/color_squares.png')

        plotColorSquares(dress_color)
        
        
        formal_shirt = pd.read_csv("C:\\Users\\ADMIN\OneDrive\\Desktop\\fashion_project\\uploads\\men-formal-shirts.csv", index_col = 0)
        def recommend(tone):
            formal_color = formal_shirt['COLOUR']
            #print(formal_color)
            formal_images = formal_shirt['IMAGE']
            ls = list()
            i=0
            count=0
            #dist = list()
            if tone == 'Fair':
                print("Fair")
                for name in formal_color:
                    for i in range(1, 90):
                        if name=='Green' or name=='Blue' or name=='Black' or name=='Red' or name=='Navy':
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==90): break;

                dist = random.sample(ls, 9)

            elif tone == 'Wheatish':
                print("Wheatish")
                for name in formal_color:
                    for i in range(1, 90):
                        if name=='Green' or name=='Yellow' or name=='White' or name=='Purple' or name=='Grey' or name=='Navy' or name=='Lime Green':
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==90): break;

                dist = random.sample(ls, 9)

            elif tone == 'Medium Brown':
                print("Medium Brown")
                for name in formal_color:
                    for i in range(1, 30):
                        #print("You choose Weatish!")
                        if name=='Light Grey' or name=='Blue' or name=='Lavender' or name=='Green' or name=='Purple' or name=='Cream':
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==30): break;
                #print(len(ls))
                dist = random.sample(ls, 9)

            elif tone == 'Brown':
                print("Brown")
                for name in formal_color:
                    for i in range(1, 90):
                        if name=='Burgundy' or name=='Dark Grey' or name=='Green' or name=='Maroon' or name=='Navy Blue' or name=='White':
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==90): break;

                dist = random.sample(ls, 9)

            elif tone == 'Dark Brown':
                print("Dark Brown")
                for name in formal_color:
                    for i in range(1, 30):
                        #print("You choose Weatish!")
                        if name=='White' or name=='Green' or name=='Purple' or name=='Dark Grey' or name=='Mustard' or name=='Coffee Brown' or name=='Charcoal Grey' :
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==30): break;

                dist = random.sample(ls, 9)
                
            else :
                print("Other")
                for name in formal_color:
                    for i in range(1, 30):
                        #print("You choose Weatish!")
                        if name=='Baby Blue' or name=='Green' or name=='Cream' or name=='Sky Blue' or name=='Red White' or name=='Pink' or name=='Blue':
                            ls.append(formal_images[i])
                        i = i+1
                    if(i==30): break;

                dist = random.sample(ls, 9)

            return dist
        sk_tone = skin_tone
        outfit = recommend(sk_tone)
        

        w = 10
        h = 10
        fig = plt.figure(figsize=(14, 12))
        columns = 4
        rows = 2
        for i in range(1, columns*rows +1):
            j = outfit[i]
            img = io.imread(j)
            fig.add_subplot(rows, columns, i)
            
        
        def display_images(image_list, count=10):
            for i in range(count):
                try:
                    image_url = image_list.iloc[i]
                    image = io.imread(image_url)
                    
                except IndexError:
                    print(f"Index {i} is out of bounds for the image list.")
                except Exception as e:
                    print(f"An error occurred: {e}")
                    
        # Add these lines in the predict function

        formal_shirts = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion_project\\uploads\\men-formal-shirts.csv", index_col=0)
        formal_shirts_images = formal_shirts['IMAGE']

        formal_trouser = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion_project\\uploads\\men-formal-trousers(1).csv", index_col=0)
        trouser_images = formal_trouser['IMAGE']

        suits = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion_project\\uploads\\men-suits.csv", index_col=0)
        suits_images = suits['IMAGE']

        casuals = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion_project\\uploads\\men-casual-shirts.csv", index_col=0)
        casuals_images = casuals['IMAGE']

        jeans = pd.read_csv("C:\\Users\\ADMIN\\OneDrive\\Desktop\\fashion_project\\uploads\\men-jeans.csv", index_col=0)
        jeans_images = jeans['IMAGE']

    
        # Get recommendations from CSV file
        csv_path = "C:\\Users\\ADMIN\\OneDrive\\Desktop\\detect.csv"
        recommendations = get_recommendations(csv_path, predicted_class, skin_tone, body_shape)
        
        return render_template('result.html', predicted_class=predicted_class, skin_tone=skin_tone, recommendations=recommendations, formal_trouser_images=trouser_images, suits_images=suits_images, casuals_images=casuals_images, jeans_images=jeans_images, formal_shirts_images=formal_shirts_images)
        #return render_template('result.html', predicted_class=predicted_class, skin_tone=skin_tone, recommendations=recommendations)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)