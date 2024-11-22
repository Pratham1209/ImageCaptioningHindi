import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from PIL import Image
from tensorflow.keras.layers import LSTM

# Define a custom LSTM to ignore unsupported arguments
class CustomLSTM(LSTM):
    def _init_(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Ignore the time_major argument
        super()._init_(*args, **kwargs)

# File paths (Change this to your actual paths)
MODEL_PATH = "/workspaces/ImageCaptioningHindi/Attention.h5"
# MODEL_PATH = "/workspaces/ImageCaptioningHindi/WithoutAttentionModel.h5"
CAPTIONS_PATH = "/workspaces/ImageCaptioningHindi/hindi_captions.txt"
TOKENIZER_PATH = "/workspaces/ImageCaptioningHindi/tokenizer.pkl"

# Load the trained model
capmodel = load_model(MODEL_PATH,custom_objects={"LSTM": CustomLSTM})

# Load captions and tokenizer from local files
def load_captions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        captions = file.readlines()
    captions_dict = {}
    for line in captions:
        image, caption = line.split('\t')
        image = image.split('#')[0].strip()
        caption = caption.strip()
        caption = f"<start> {caption} <end>"
        if image not in captions_dict:
            captions_dict[image] = []
        captions_dict[image].append(caption)
    return captions_dict

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return tokenizer

# Initialize captions and tokenizer
captions_dict = load_captions(CAPTIONS_PATH)
tokenizer = load_tokenizer(TOKENIZER_PATH)

# Extract features from an image using InceptionV3
def extract_features(img):
    model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    img = img.resize((299, 299))  # Resize to match InceptionV3 input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

# Generate caption
def generate_caption(model, tokenizer, features, max_length=47):
    caption = "<start>"
    consecutive_end_count = 0  # Track consecutive "end" tokens
    result_caption = []

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        prediction = model.predict([features, sequence], verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_word = tokenizer.index_word.get(predicted_index, "<unknown>")

        if predicted_word == "end":
            consecutive_end_count += 1
            if consecutive_end_count >= 1:  # Stop if "end" appears consecutively
                break
        else:
            consecutive_end_count = 0  # Reset if a valid word is generated

        result_caption.append(predicted_word)
        caption += " " + predicted_word

    # Remove any <start> or <end> tokens from the final caption
    return " ".join(word for word in result_caption if word not in ["<start>", "<end>"])


# Streamlit UI
st.title("Image Caption Generator in Hindi")
st.write("Upload an image to generate a caption in Hindi.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Extract features and generate caption
    test_features = extract_features(img)
    caption = generate_caption(capmodel, tokenizer, test_features)
    
    # Display the generated caption
    st.write("Generated Caption: ")
    st.markdown(f"**{caption}**")
