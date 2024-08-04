import streamlit as st
import os
from PIL import Image
import io
from similarity import list_top_similar
import utils as utl

st.title('Image Similarity Search')
model_options = {"CLIP": "clip", "ViT": "vit"}
model_name = st.radio("Choose a model for the similarity search:", list(model_options.keys()))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
utl.make_dir("temp")
temp_path = "temp/query_image.jpg"
if uploaded_file is not None:
    image_bytes = io.BytesIO(uploaded_file.getvalue())
    ref_image = Image.open(image_bytes)

    # Display the uploaded image
    st.image(ref_image, caption='Uploaded Image', width=400)

    # Perform the similarity search
    similar_images = list_top_similar(ref_image, model_options[model_name], 6)
    st.write("Similar images:")

    num_columns = 3  # Adjust this value based on your preference for the grid width
    cols = st.columns(num_columns)
    index = 0
    for img_path, sim in similar_images:
        with cols[index % num_columns]:
            st.image(img_path, width=200)
            st.caption(f"{img_path} - similarity {sim:.2f}")
        index += 1

# Clean up the temporary data
if os.path.exists(temp_path):
    os.remove(temp_path)
