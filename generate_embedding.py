import os
import numpy as np
from transformers import CLIPModel, CLIPProcessor, ViTModel, ViTImageProcessor
from PIL import Image
import utils as utl

def image_embedding(image, model_name):
    #print(type(image), image.mode, image.size)
    #np_image = np.array(image)
    #print("Numpy array shape:", np_image.shape)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    model = models[model_name]
    inputs = model["processor"](images=image, return_tensors="pt")
    if model_name == "clip":
        outputs = model["model"].get_image_features(**inputs)  # This returns the proper embedding directly
        embedding_vector = outputs.detach().numpy()
    elif model_name == "vit":
        outputs = model["model"](inputs['pixel_values'])[0]  # Assuming this is the correct index for the output tensor
        embedding_vector = outputs.mean(dim=1).detach().numpy()  # Average pooling over the sequence dimension
    return embedding_vector

def collect_images(folder_path):
    print("collecting images")
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg'))]
    print(f"found {len(images)} image")
    for i,image_path in enumerate(images):
        images[i] = image_path.replace("\\","/")
    utl.save_json(images,"data/images.json")
    return images

def generate_images_embedding(images,model_name):
    print(f"Geenrating embeddings for '{model_name}'")
    embeddings = {}
    for image_path in images:
        image = Image.open(image_path)
        print(f"Generating Embedding Vector for {image_path}:")
        embedding = image_embedding(image,model_name)
        embeddings[image_path] = embedding.tolist()
    utl.save_json(embeddings,f"data/embeddings-{model_name}.json")

print("creating models (clip, vit)")
models = {
    "clip": {
        "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
        "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    },
    "vit": {
        "model": ViTModel.from_pretrained("google/vit-base-patch16-224"),
        "processor": ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    }
}

if __name__ == "__main__":
    images = collect_images("images")
    generate_images_embedding(images,"clip")
    generate_images_embedding(images,"vit")
