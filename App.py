import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Sidebar
st.sidebar.title("🧠 Settings")
st.sidebar.write("Configure the model and caption settings:")

model_file = st.sidebar.selectbox("Select Model File", ["caption_model.keras"])
max_length = st.sidebar.slider("Max Caption Length", min_value=10, max_value=60, value=38)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 *Trained on Flickr8k*")
st.sidebar.markdown("👤 Developed by [You]")

#Load model, tokenizer
model = tf.keras.models.load_model(model_file, compile=False)

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

#Load VGG16 for feature extraction 
vgg_model = VGG16()
vgg_model = tf.keras.models.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('fc1').output)

# Generate caption
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)

        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = next((w for w, index in tokenizer.word_index.items() if index == yhat), None)
        if word is None:
            break

        in_text += ' ' + word
        if word == 'endseq':
            break

    return in_text.replace('startseq', '').replace('endseq', '').strip()

#Extract features from uploaded image
def extract_features(img):
    img = img.resize((224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    feature = vgg_model.predict(img, verbose=0)
    return feature

# Main App 
st.title("📸 Image Caption Generator")
st.write("Upload an image and let the AI describe it for you.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Generating caption..."):
        feature = extract_features(image)
        caption = generate_caption(model, tokenizer, feature, max_length)

    st.success(" Caption Generated!")
    st.markdown(f"**📝 Caption:** _{caption}_")
