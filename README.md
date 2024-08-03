# Overview
This repo is about exploring image search techniques.

First test is using torch and openai `clip-vit-base-patch32`

The demo was made with `streamlit`

test images from
- https://www.homesmartmesh.com/
- https://github.com/HomeSmartMesh/website/tree/main/public/images

# Demo
![drop similarity](./images-similarity.gif)

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
