import streamlit as st
import numpy as np
import os
from PIL import Image
import images_features  # Assuming this contains the image_to_embedding function
import similarity  # Assuming this contains the list_top_similar function
import utils as utl

# Load pre-computed embeddings
embeddings = utl.load_json("data/embeddings.json")
for key in embeddings:
    embeddings[key] = np.array(embeddings[key])

st.title('Image Similarity Search')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
utl.make_dir("temp")
temp_path = "temp/query_image.jpg"
if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', width=400)
    # Save the query embedding temporarily to use in similarity search
    print(f"uploaded file to '{temp_path}'")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    query_embedding = images_features.image_to_embedding(temp_path)
    embeddings[temp_path] = query_embedding

    # Perform the similarity search
    similar_images = similarity.list_top_similar(temp_path, embeddings, 5)
    st.write("Similar images:")

    # Display similar images
    for img_path, _ in similar_images:
        st.caption(img_path)
        st.image(img_path, width=200)

# Clean up the temporary data
if os.path.exists(temp_path):
    os.remove(temp_path)
    del embeddings[temp_path]

