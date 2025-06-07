import numpy as np
import pickle
import csv
import string
from collections import defaultdict
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.utils import Sequence
import random
import tensorflow as tf

#STEP 1: Load preprocessed data ===

# Load image features
features = np.load('image_features_vgg.npy', allow_pickle=True).item()

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load and clean captions
captions = defaultdict(list)
with open('C:/Users/abish/Desktop/Guvi_Files/Project_Final/archive/Text/captions.txt', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        image = row['image'].strip()
        caption = row['caption'].strip().lower()
        caption = caption.translate(str.maketrans('', '', string.punctuation))
        caption = 'startseq ' + caption + ' endseq'
        captions[image].append(caption)

# Create list of keys (image names)
keys = list(captions.keys())

# Set max length and vocab size (should match training)
max_length = max(len(c.split()) for c in [cap for sublist in captions.values() for cap in sublist])
vocab_size = len(tokenizer.word_index) + 1

#STEP 2: Define Data Generator ===

class DataGenerator(Sequence):
    def __init__(self, keys, captions, features, tokenizer, max_length, vocab_size, batch_size=32):
        self.keys = keys
        self.captions = captions
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.keys) / float(self.batch_size)))

    def __getitem__(self, index):
        X1, X2, y = [], [], []

        keys = self.keys[index * self.batch_size:(index + 1) * self.batch_size]
        for key in keys:
            caption = random.choice(self.captions[key])  # use one caption per image
            seq = self.tokenizer.texts_to_sequences([caption])[0]

            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]

                X1.append(self.features[key])
                X2.append(in_seq)
                y.append(out_seq)

        return (np.array(X1), np.array(X2)), np.array(y)

    def on_epoch_end(self):
        random.shuffle(self.keys)

#STEP 3: Build the Model ===

# Image feature input
inputs1 = Input(shape=(4096,))
x1 = Dropout(0.5)(inputs1)
x1 = Dense(256, activation='relu')(x1)

# Caption sequence input
inputs2 = Input(shape=(max_length,))
x2 = Embedding(input_dim=vocab_size, output_dim=256)(inputs2)
x2 = Dropout(0.5)(x2)
x2 = LSTM(256)(x2)

# Merge and output
decoder1 = Add()([x1, x2])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()

#STEP 4: Train Using Generator ===

generator = DataGenerator(
    keys=keys,
    captions=captions,
    features=features,
    tokenizer=tokenizer,
    max_length=max_length,
    vocab_size=vocab_size,
    batch_size=32
)

model.fit(generator, epochs=10)

#STEP 5: Save the Model ===

model.save('caption_model.keras', save_format='keras')
print("\nModel trained and saved as 'caption_model.keras'")
