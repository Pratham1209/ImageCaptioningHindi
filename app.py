import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from tensorflow.keras.layers import LSTM


class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        # Safely remove "time_major" argument, if exists
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)

# File paths
MODEL_PATHS = {
    "Without Attention": "WithoutAttentionModel.h5",
    "With Attention": "Attention.h5",
    "Transformer-based Captioning": "MHA.h5",
    "BERT-based Captioning": "BERT.h5",
}
CAPTIONS_PATH = "hindi_captions.txt"

# Load captions and initialize tokenizer
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

def initialize_tokenizer(captions_dict):
    all_captions = [cap for caps in captions_dict.values() for cap in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_captions)
    return tokenizer

captions_dict = load_captions(CAPTIONS_PATH)
tokenizer = initialize_tokenizer(captions_dict)

# Extract features from an image using InceptionV3
def extract_features(img):
    model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    img = img.resize((299, 299))  
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return model.predict(img)

# Generate caption
def generate_caption(model, tokenizer, features, max_length=47):
    caption = "<start>"
    consecutive_end_count = 0  # Track consecutive "end" token
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

    # Remove any <start> or <end> tokens from the final captio
    return " ".join(word for word in result_caption if word not in ["<start>", "<end>"])

# Streamlit UI
st.set_page_config(page_title="Hindi Image Caption Generator", page_icon="📸")

# CSS
st.markdown(
    """
    <style>
        .main-title {
            font-size: 35px;
            font-weight: bold;
            color: #2E86C1;
            text-align: center;
        }
        .sub-title {
            font-size: 20px;
            color: #566573;
            text-align: center;
        }
        .footer {
            font-size: 14px;
            text-align: center;
            color: #ABB2B9;
            margin-top: 50px;
        }
        .button {
            text-align: left;
            font-size: 18px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<div class="main-title">Image Caption Generator in Hindi 📸</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image and select a model to generate captions in Hindi</div>', unsafe_allow_html=True)


with st.sidebar:
    st.button("Developers", key="menu", help="Click to see options")
    st.markdown("**Names:**\n1. Maitreyee Deshmukh\n2. Pratham Patharkar\n3. Mohit Lalwani\n4. Tanavi Gaikwad", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### About the Project")
    
    about_english = """
    ### Why Hindi Captioning?
    India is a diverse country where Hindi is the most spoken language, yet most AI solutions focus only on English, leaving millions excluded. Hindi captioning bridges this gap by:
    
    - **Promoting inclusivity** for non-English speakers.
    - **Fostering accessibility** for visually impaired users with Hindi descriptions.
    - **Representing culture** through technology in native languages.
    
    This ensures AI is for everyone, regardless of language barriers.
    """
    
    about_hindi = """
    ### हिंदी कैप्शनिंग क्यों?
    भारत में हिंदी सबसे अधिक बोली जाने वाली भाषा है, लेकिन अधिकतर एआई समाधान केवल अंग्रेजी तक सीमित हैं। हिंदी कैप्शनिंग इस अंतर को कम करती है:
    
    - **सभी को शामिल करना** जो अंग्रेजी नहीं बोलते।
    - **सुलभता बढ़ाना** दृष्टिहीन उपयोगकर्ताओं के लिए।
    - **संस्कृति का प्रतिनिधित्व करना**।
    
    यह सुनिश्चित करता है कि एआई हर किसी के लिए हो।
    """
    
    language = st.radio("Select Language | भाषा चुनें", ["English", "हिंदी"])
    if language == "English":
        st.markdown(about_english)
    else:
        st.markdown(about_hindi)


selected_model = st.selectbox(
    "Select a model for caption generation:",
    options=list(MODEL_PATHS.keys())
)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and selected_model:
    try:
        # Display the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        # Load the selected model
        model_path = MODEL_PATHS[selected_model]
        st.write(f"Loading model: {selected_model}")
        capmodel = load_model(model_path, custom_objects={"LSTM": CustomLSTM})
        
        # Extract features and generate caption
        test_features = extract_features(img)
        caption = generate_caption(capmodel, tokenizer, test_features)
        
        # Display the generated caption
        st.write("Generated Caption: ")
        st.markdown(f"**{caption}**")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown('<div class="footer">Created with ❤️ by Team Enigma</div>', unsafe_allow_html=True)
