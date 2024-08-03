import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import utils as utl

def image_to_embedding(image_path):
    # Load the pre-trained CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt", padding=True)
    outputs = model.get_image_features(**inputs)
    embedding_vector = outputs.detach().numpy()

    return embedding_vector

def process_images_in_folder(folder_path):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    utl.save_json(images,"data/images.json")
    # Process each image and store/print embeddings
    embeddings = {}
    for image_path in images:
        embedding = image_to_embedding(image_path)
        print(f"Generated Embedding Vector for {image_path}:")
        embeddings[image_path] = embedding.tolist()
    utl.save_json(embeddings,"data/embeddings.json")

if __name__ == "__main__":
    process_images_in_folder("images")
