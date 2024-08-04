import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import utils as utl

def image_embedding_clip(image,model):
    inputs  = model["processor"](images=image, return_tensors="pt", padding=True)
    outputs = model["model"].get_image_features(**inputs)
    embedding_vector = outputs.detach().numpy()
    return embedding_vector

def generate_images_embedding(folder_path,model_name):
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    for image_path in images:
        image_path = image_path.replace("\\","/")
    utl.save_json(images,"data/images.json")
    # Process each image and store/print embeddings
    embeddings = {}
    for image_path in images:
        image = Image.open(image_path)
        embedding = image_embedding_clip(image,models[model_name])
        print(f"Generated Embedding Vector for {image_path}:")
        embeddings[image_path] = embedding.tolist()
    utl.save_json(embeddings,f"data/embeddings-{model_name}.json")

clip = {
    "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
}
models = {"clip":clip}

if __name__ == "__main__":
    generate_images_embedding("images","clip")
