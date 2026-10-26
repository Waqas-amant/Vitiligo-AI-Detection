# Vitiligo AI Detection

A skin-image classification web app using a pretrained Vision Transformer (ViT) model and Flask.

## Features
- Upload a skin image for prediction
- Capture an image using the browser camera
- Display predicted class and model confidence
- Model evaluation: accuracy, precision, recall, F1-score, and confusion matrix

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Vision Transformer (ViT)
- Flask
- HTML, CSS, JavaScript

## Run Locally
1. Install Python dependencies.
2. Place the trained model files in the `model/` folder.
3. Run `python app.py`.
4. Open `http://127.0.0.1:5001` in your browser.

## Disclaimer
This is an educational prototype, not a medical diagnostic tool. Consult a qualified dermatologist for medical assessment.
