import streamlit as st
import cv2
import numpy as np
import easyocr
from googletrans import Translator
from PIL import Image

# Initialize the EasyOCR Reader
# We use st.cache_resource so it doesn't reload every time you click a button
@st.cache_resource
def load_reader():
    # 'gu' = Gujarati, 'hi' = Hindi/Sanskrit, 'en' = English
    return easyocr.Reader(['gu', 'hi', 'en'], gpu=False)

st.set_page_config(page_title="Vision OCR & Translator")
st.title("📸 OCR Text Extractor & Translator")

# --- Step 1: Upload Image ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Language Selection
target_lang = st.selectbox(
    "Translate to:",
    options=["en", "gu", "hi", "sa"],
    format_func=lambda x: {"en": "English", "gu": "Gujarati", "hi": "Hindi", "sa": "Sanskrit"}[x]
)

if uploaded_file is not None:
    # --- Step 2: Use OpenCV to process image ---
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Show the image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_container_width=True)
    
    if st.button('Extract & Translate'):
        with st.spinner('Processing...'):
            try:
                # 3. OCR Detection
                reader = load_reader()
                # Convert to grayscale for better detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                results = reader.readtext(gray, detail=0)
                full_text = " ".join(results)

                if full_text.strip():
                    st.subheader("Detected Text:")
                    st.info(full_text)

                    # 4. Translation
                    translator = Translator()
                    translated = translator.translate(full_text, dest=target_lang)
                    
                    st.subheader(f"Translated Text ({target_lang}):")
                    st.success(translated.text)
                else:
                    st.warning("No text found.")
            except Exception as e:
                st.error(f"Error: {e}")
                
