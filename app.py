import streamlit as st
import os
from PIL import Image
import io
import similarity  # Assuming this contains the list_top_similar function
import utils as utl

st.title('Image Similarity Search')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
utl.make_dir("temp")
temp_path = "temp/query_image.jpg"
if uploaded_file is not None:
    image_bytes = io.BytesIO(uploaded_file.getvalue())
    ref_image = Image.open(image_bytes)

    # Display the uploaded image
    st.image(ref_image, caption='Uploaded Image', width=400)

    # Perform the similarity search
    similar_images = similarity.list_top_similar(ref_image, "clip", 5)
    st.write("Similar images:")

    # Display similar images
    for img_path, sim in similar_images:
        st.caption(f"{img_path} - similarity {sim:.2f}")
        st.image(img_path, width=200)

# Clean up the temporary data
if os.path.exists(temp_path):
    os.remove(temp_path)
