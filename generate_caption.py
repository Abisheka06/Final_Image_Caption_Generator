import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Layer
import tensorflow as tf
model = tf.keras.models.load_model('caption_model.keras', compile=False)

#STEP 1: Load tokenizer, model, and image features ===

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load image features
features = np.load('image_features_vgg.npy', allow_pickle=True).item()

# Get max caption length (use a fixed number or store from training)
max_length = 38  # adjust based on your model_building.py output

#STEP 2: Generate caption for an image ===

def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break

        if word is None:
            break
        in_text += ' ' + word

        if word == 'endseq':
            break

    # Clean final output
    final = in_text.replace('startseq', '').replace('endseq', '').strip()
    return final

#STEP 3: Choose and caption an image ===

# Choose any image from the features
image_id = '1001773457_577c3a7d70.jpg'  # You can change this to another image name
photo = features[image_id].reshape((1, 4096))  # VGG16 feature size

caption = generate_caption(model, tokenizer, photo, max_length)
print(f"Image: {image_id}")
print(f"Caption: {caption}")
