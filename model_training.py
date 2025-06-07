
import numpy as np
import string
import pickle
import csv
from collections import defaultdict

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from model_cnn_rnn import model, max_length, vocab_size  # Reuse model from Day 3

#STEP 1: Load image features ===
features = np.load('image_features_vgg.npy', allow_pickle=True).item()

#STEP 2: Load tokenizer ===
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

#STEP 3: Load and clean captions ===
captions = defaultdict(list)

with open('C:/Users/abish/Desktop/Guvi_Files/Project_Final/archive/Text/captions.txt', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image = row['image'].strip()
        caption = row['caption'].strip().lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        caption = 'startseq ' + caption + ' endseq'
        captions[image].append(caption)

#STEP 4: Create training data ===

def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = [], [], []

    for key in list(descriptions.keys())[:200]:  # LIMIT to 200 images
        desc_list = descriptions[key]
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]

                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                X1.append(photos[key])
                X2.append(in_seq)
                y.append(out_seq)

    return np.array(X1), np.array(X2), np.array(y)

print("Creating training data...")

X1, X2, y = create_sequences(tokenizer, max_length, captions, features, vocab_size)

print(" Training data shapes:")
print("X1 (image features):", X1.shape)
print("X2 (input text sequences):", X2.shape)
print("y (next word):", y.shape)

# === STEP 5: Train the model ===

print("Training the model (this will take time)...")

model.fit([X1, X2], y, epochs=5, batch_size=16)

#STEP 6: Save the trained model ===

model.save('caption_model.keras', save_format='keras')

print(" Model saved as caption_model.keras")
