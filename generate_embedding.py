import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import utils as utl

def image_embedding(image,model_name):
    model = models[model_name]
    inputs  = model["processor"](images=image, return_tensors="pt", padding=True)
    outputs = model["model"].get_image_features(**inputs)
    embedding_vector = outputs.detach().numpy()
    return embedding_vector

def generate_images_embedding(folder_path,model_name):
    print("collecting images")
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    print(f"found {len(images)} image")
    for i,image_path in enumerate(images):
        images[i] = image_path.replace("\\","/")
    utl.save_json(images,"data/images.json")
    print("Geenrating embedding for each image")
    embeddings = {}
    for image_path in images:
        image = Image.open(image_path)
        embedding = image_embedding(image,model_name)
        print(f"Generated Embedding Vector for {image_path}:")
        embeddings[image_path] = embedding.tolist()
    utl.save_json(embeddings,f"data/embeddings-{model_name}.json")

print("creating CLIP model")
clip = {
    "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
    "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
}
models = {"clip":clip}

if __name__ == "__main__":
    generate_images_embedding("images","clip")
