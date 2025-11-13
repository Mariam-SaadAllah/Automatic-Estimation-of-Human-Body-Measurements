import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import gradio as gr
from pathlib import Path
import numpy as np

from bodym.model import MNASNetRegressor
from bodym.data import Y_MIN_MM, Y_MAX_MM, MEASUREMENT_COLS

# === Load trained model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = Path("/content/drive/MyDrive/BMNet_Project/checkpoints/best_mnasnet_bmnet.pt")

model = MNASNetRegressor(num_outputs=14, weights=None)
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.to(device)
model.eval()

# === Define preprocessing (same as during training) ===
preprocess = transforms.Compose([
    transforms.Resize((640, 480)),  # same as training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # if grayscale silhouette
])

# === Prediction function for Gradio ===
def predict_body_measurements(image):
    # Convert uploaded image to model input
    if image.mode != "L":  # convert to grayscale if needed
        image = image.convert("L")

    x = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred = model(x)

    # De-normalize predictions back to millimeters
    y_mm = ((y_pred + 1) / 2) * (torch.from_numpy(Y_MAX_MM).to(device) - torch.from_numpy(Y_MIN_MM).to(device)) + torch.from_numpy(Y_MIN_MM).to(device)
    y_mm = y_mm.cpu().numpy().flatten()

    # Prepare a results dictionary
    results = {name: f"{val:.2f} mm" for name, val in zip(MEASUREMENT_COLS, y_mm)}
    return results

# === Gradio UI ===
demo = gr.Interface(
    fn=predict_body_measurements,
    inputs=gr.Image(label="Upload Silhouette", type="pil"),
    outputs=gr.JSON(label="Predicted Body Measurements (mm)"),
    title="Human Body Measurement Estimation",
    description="Upload a single-view silhouette to estimate 14 human body measurements using the trained MNASNet model."
)

demo.launch(share=True)
