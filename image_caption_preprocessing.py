# image_preprocessing.py

import os
import numpy as np
import string
import csv
from collections import defaultdict
import tensorflow as tf

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model

#STEP 1: LOAD CAPTIONS
captions = defaultdict(list)

with open('C:/Users/abish/Desktop/Guvi_Files/Project_Final/archive/Text/captions.txt', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image = row['image'].strip()
        caption = row['caption'].strip()
        captions[image].append(caption)

print(f"✅ Captions loaded for {len(captions)} images.")

#STEP 2: CLEAN CAPTIONS
def clean_caption(caption):
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.strip()
    caption = 'startseq ' + caption + ' endseq'
    return caption

for img in captions:
    captions[img] = [clean_caption(c) for c in captions[img]]

print("✅ Captions cleaned.")

#STEP 3: LOAD VGG16 MODEL FOR FEATURE EXTRACTION
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

#STEP 4: IMAGE PROCESSING FUNCTION
def process_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # VGG16 input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

#STEP 5: EXTRACT FEATURES FOR ALL IMAGES
image_folder = 'C:/Users/abish/Desktop/Guvi_Files/Project_Final/archive/Images'
features = {}

for img_name in os.listdir(image_folder):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(image_folder, img_name)
        img = process_image(img_path)
        feature = model.predict(img, verbose=0)
        features[img_name] = feature.flatten()

# Save extracted features
np.save('image_features_vgg.npy', features)
print(" VGG16 image features saved as image_features_vgg.npy")
