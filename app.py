
import os
import torch
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request
from transformers import ViTForImageClassification, ViTImageProcessor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
model = ViTForImageClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("image")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please select an image first."
        else:
            try:
                image = Image.open(uploaded_file.stream).convert("RGB")

                inputs = processor(images=image, return_tensors="pt")
                inputs = {key: value.to(device)
                          for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.softmax(outputs.logits, dim=1)[0]

                predicted_id = int(probabilities.argmax().item())
                confidence = round(
                    float(probabilities[predicted_id].item()) * 100, 2
                )

                result = model.config.id2label.get(
                    predicted_id, str(predicted_id)
                )

            except (UnidentifiedImageError, OSError):
                error = "The uploaded file is not a valid image."
            except Exception:
                app.logger.exception("Prediction failed")
                error = "Prediction failed. Please try another image."

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        error=error
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
