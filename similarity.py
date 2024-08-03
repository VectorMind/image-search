import os
import utils as utl
import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec_a, vec_b):
    """Calculate the cosine similarity between two vectors."""
    vec_a = vec_a.flatten()
    vec_b = vec_b.flatten()
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b))

def list_top_similar(ref_image_path, image_embedding_map,k):
    ref_embedding = image_embedding_map[ref_image_path]

    # Calculate similarity of the reference image with all other images
    similarities = {}
    for image_path, embedding in image_embedding_map.items():
        # Calculate cosine similarity and store it
        sim = cosine_similarity(ref_embedding, embedding)
        similarities[image_path] = sim

    # Sort images by similarity (higher first)
    sorted_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    return sorted_images[:k]


embeddings = utl.load_json("data/embeddings.json")
for key in embeddings:
    embeddings[key] = np.array(embeddings[key])


if __name__ == "__main__":
    ref_image = "images\\stm32_bluepill.jpg"

    similar = list_top_similar(ref_image, embeddings,5)
    print(f"similar images to '{ref_image}' are:")
    print(similar)
    result = {
        "ref":ref_image,
        "similar":similar
    }
    filename = os.path.splitext(os.path.basename(ref_image))[0]
    utl.save_json(result,f"data/{filename}.json")
