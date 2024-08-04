import os
import utils as utl
import numpy as np
from numpy.linalg import norm
from PIL import Image
from generate_embedding import image_embedding

def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def list_top_similar(ref_image, model_name,k):
    ref_embedding = image_embedding(ref_image,"clip")
    model_embeddings_map = embeddings_map[model_name]
    # Calculate similarity of the reference image with all other images
    similarities = {}
    for image_path, embedding in model_embeddings_map.items():
        # Calculate cosine similarity and store it
        sim = cosine_similarity(ref_embedding, embedding)
        similarities[image_path] = sim

    # Sort images by similarity (higher first)
    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_images[:k]

def load_model_embeddings(model_name):
    print(f"loading '{model_name}' embeddings")
    embeddings = utl.load_json(f"data/embeddings-{model_name}.json")
    for key in embeddings:
        embeddings[key] = np.array(embeddings[key])
    return embeddings

clip_embeddings = load_model_embeddings("clip")

embeddings_map = {"clip":clip_embeddings}

if __name__ == "__main__":
    ref_image_path = "images/stm32_bluepill.jpg"
    ref_image = Image.open(ref_image_path)
    similar = list_top_similar(ref_image, "clip",5)
    print(f"similar images to '{ref_image_path}' are:")
    print(similar)
    result = {
        "ref":ref_image_path,
        "similar":similar
    }
    filename = os.path.splitext(os.path.basename(ref_image_path))[0]
    utl.save_json(result,f"data/{filename}.json")
