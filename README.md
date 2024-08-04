# Overview
Exploring image search techniques using transformer models.

# Demo
The demo was made with `streamlit`
![drop similarity](./images-similarity-clip-vit.gif)

# Test
Models :
* CLIP : `openai/clip-vit-base-patch32`
* ViT : `google/vit-base-patch16-224`

## Test images
The test images are available here
- https://www.homesmartmesh.com/
- https://github.com/HomeSmartMesh/website/tree/main/public/images

Two images have been cropped and the cropped images have been input for search in the full images dataset

* home model

Full image

![full](./test-images/full-image.png)

Cropped image

![cropped](./test-images/crop.png)

* Savana Scene

Full image

![full scene](./test-images/full-scene.jpg)

Test image

![test scene](./test-images/scene.png)

# Results
|Test image |model | hit position | similarity|
|-----------|------|--------------|-----------|
| home model|ViT | 7th  |0.39 |
| home model|CLIP | 96th |  0.44 |
| savana scene|ViT | 1st  |0.68 |
| savana scene|CLIP | 3rd |  0.15 |

## Concepts
* Content-Based Image Retrieval (CBIR)
* Image Recognition
* Feature Extraction
* Semantic Search using trnasformers
## Tools and libraries
* OpenCV
* Scikit-Image
* Pillow/PIL
* TensorFlow and PyTorch
* Elasticsearch and other search engines like typesense

## Image similarities
* https://huggingface.co/blog/image-similarity

## Image based transformers
* CLIP : Contrastive Language–Image Pre-training

# Setup
transformers require PyTorch to be installed
* https://pytorch.org/get-started/locally/

```cmd
pip3 install torch torchvision torchaudio
```
